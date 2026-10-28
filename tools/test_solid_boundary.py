import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import manifold3d as m
import solid_boundary
from solid_boundary import bake

SCRIPT = Path(solid_boundary.__file__).resolve()


def box(low, high, material=0):
    mesh=m.Manifold.cube(tuple(np.array(high)-low)).translate(tuple(low)).to_mesh64()
    return dict(vertices=np.asarray(mesh.vert_properties).ravel().tolist(),
                triangles=np.asarray(mesh.tri_verts).ravel().tolist(),
                materials=[material]*len(mesh.tri_verts))


def run(*bodies): return bake(dict(version=1,bodies=list(bodies)))


def sharp_edges(data):
    v=np.array(data['vertices']).reshape(-1,3);t=np.array(data['triangles']).reshape(-1,3)
    normals=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    uses={}
    for i,tri in enumerate(t):
        for a,b in zip(tri,np.roll(tri,-1)):uses.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    return {edge for edge,faces in uses.items() if len(faces)!=2 or np.dot(normals[faces[0]],normals[faces[1]])<.99999}


class SolidBoundaryTests(unittest.TestCase):
    def test_touching_and_overlapping_partitions_have_the_same_exterior(self):
        reference=run(box([-1]*3,[1]*3))
        for seam,overlap in [(-.8,0),(0,0),(.4,.3)]:
            parts=[box([-1]*3,[seam,1,1]),box([seam-overlap,-1,-1],[1]*3)]
            for bodies in [parts,parts[::-1]]:
                actual=run(*bodies)
                self.assertEqual(reference['vertices'],actual['vertices'])
                self.assertEqual(sharp_edges(reference),sharp_edges(actual))
                self.assertEqual(set(actual['materials']),{0})
                self.assertAlmostEqual(actual['volume'],8);self.assertAlmostEqual(actual['area'],24)
                self.assertEqual(actual['outputTriangles'],12)

    def test_gaps_survive(self):
        actual=run(box([-1]*3,[-.8,1,1]),box([-.7,-1,-1],[1]*3))
        self.assertEqual(actual['components'],2)
        self.assertAlmostEqual(actual['volume'],7.6)
        self.assertAlmostEqual(actual['area'],31.2)

    def test_surviving_faces_keep_materials(self):
        actual=run(box([-1]*3,[0,1,1],3),box([0,-1,-1],[1]*3,7))
        v=np.array(actual['vertices']).reshape(-1,3);t=np.array(actual['triangles']).reshape(-1,3)
        a=np.array(actual['materials']);centers=v[t].mean(axis=1)
        self.assertEqual(set(a),{3,7})
        self.assertTrue(np.all(a[centers[:,0]<-1e-8]==3));self.assertTrue(np.all(a[centers[:,0]>1e-8]==7))
        self.assertFalse(np.any(np.all(np.abs(v[t][:,:,0])<1e-8,axis=1)), 'No internal x=0 face.')

    def test_member_order_resolves_coincident_surface_materials(self):
        first=box([-1]*3,[.5,1,1],3);second=box([0,-1,-1],[1]*3,7)
        for bodies,seam in [([first,second],.5),([second,first],0)]:
            actual=run(*bodies);v=np.array(actual['vertices']).reshape(-1,3);t=np.array(actual['triangles']).reshape(-1,3)
            x=v[t].mean(axis=1)[:,0];labels=np.array(actual['materials'])
            self.assertTrue(np.all(labels[x<seam-1e-8]==3));self.assertTrue(np.all(labels[x>seam+1e-8]==7))
            self.assertAlmostEqual(actual['volume'],8);self.assertAlmostEqual(actual['area'],24)
        same=run(box([-1]*3,[1]*3,3),box([-1]*3,[1]*3,7))
        self.assertEqual(set(same['materials']),{3})

    def test_reflected_winding_and_exact_vertex_duplicates(self):
        body=box([-1]*3,[1]*3)
        v=np.array(body['vertices']).reshape(-1,3);t=np.array(body['triangles']).reshape(-1,3)
        xyz=v[t][:,::-1,:];body['vertices']=xyz.ravel().tolist();body['triangles']=list(range(len(xyz)*3))
        actual=run(body);self.assertAlmostEqual(actual['volume'],8);self.assertEqual(actual['outputTriangles'],12)

    def test_open_or_invalid_mesh_is_rejected(self):
        body=box([-1]*3,[1]*3);body['triangles']=body['triangles'][:-3];body['materials']=body['materials'][:-1]
        with self.assertRaises(ValueError):run(body)
        body=box([-1]*3,[1]*3);body['vertices'][0]=float('nan')
        with self.assertRaises(ValueError):run(body)



class CommandLineTests(unittest.TestCase):
    """The Unity Editor relies on these exit codes and messages."""
    def call(self,spec,prelude=''):
        folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        source,output=folder/'input.json',folder/'output.json'
        source.write_text(json.dumps(spec))
        code=(prelude+'import runpy,sys;sys.argv=[sys.argv[1],sys.argv[2],sys.argv[3]];'
              'runpy.run_path(sys.argv[0],run_name="__main__")')
        result=subprocess.run([sys.executable,'-I','-c',code,str(SCRIPT),str(source),str(output)],capture_output=True,text=True,timeout=60)
        return result,output,source

    def test_success_writes_hash_keyed_output_atomically(self):
        result,output,source=self.call(dict(version=1,bodies=[box([-1]*3,[1]*3)]))
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(output.read_text())
        self.assertEqual(data['inputSha256'],hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(data['backend'],'manifold3d '+solid_boundary.BACKEND_VERSION)
        self.assertEqual(sorted(p.name for p in output.parent.iterdir()),['input.json','output.json'])

    def test_missing_backend_is_reported_with_install_command(self):
        result,output,_=self.call(dict(version=1,bodies=[box([-1]*3,[1]*3)]),'import sys;sys.modules["manifold3d"]=None;')
        self.assertEqual(result.returncode,3)
        self.assertIn("missing Python module 'manifold3d'",result.stderr)
        self.assertIn('-m pip install -r',result.stderr);self.assertIn('solid-boundary-requirements.txt',result.stderr)
        self.assertFalse(output.exists())

    def test_invalid_geometry_has_a_distinct_exit_code(self):
        body=box([-1]*3,[1]*3);body['triangles']=body['triangles'][:-3];body['materials']=body['materials'][:-1]
        result,output,_=self.call(dict(version=1,bodies=[body]))
        self.assertEqual(result.returncode,4);self.assertIn('Invalid solid input',result.stderr);self.assertFalse(output.exists())

    def test_backend_pin_matches_requirements(self):
        pins=solid_boundary.REQUIREMENTS.read_text().split()
        self.assertIn('manifold3d=='+solid_boundary.BACKEND_VERSION,pins)


if __name__=='__main__': unittest.main()
