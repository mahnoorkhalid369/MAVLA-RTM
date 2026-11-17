"""A side body patch must bind without pretending to observe its cap; invalid depths are not geometry."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from perception.world_model import WorldModel, ObjectRecord
from perception.capture import surface_points
from perception.instances import clean_mask
from tests.support.fakes import geom


def body_points(x=0., y=.55, yaw=0.):
    u,z=np.meshgrid(np.linspace(-.02,.02,12),np.linspace(.01,.095,20))
    points=np.column_stack([u.ravel(),np.full(u.size,-.02),z.ravel()])
    a=np.radians(yaw)
    R=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    return points @ R.T + [x,y,0.]


def world():
    w=WorldModel()
    w.observe(geom(0.,.55,.168,fp=(.074,.074)),None)
    return w


def test_actual_bottle_side_failure_binds_without_relaxing_top_view_rules():
    fixture=json.loads((Path(__file__).parents[1]/'fixtures/source_side_bottle_body.json').read_text())
    w=WorldModel();w.objects[6]=ObjectRecord(6,fixture['geometry'])
    for i,g in enumerate(fixture['other_objects'],100):
        w.objects[i]=ObjectRecord(i,g)
    bottle=fixture['marks']['1'];cracker=fixture['marks']['2']
    assert fixture['geometry']['top_z']-bottle['measured']['top_z'] > .070
    assert w.associate(bottle['measured'])[0]=='new'  # the normal top gate stays strict
    assert w.associate_partial(bottle['points'])==('matched',6,[6])
    assert w.associate_partial(cracker['points'])[1] != 6
    assert w.objects[6].geometry==fixture['geometry']  # no partial geometry update


def test_partial_view_is_unique_not_nearest_expected_object():
    w=world();points=body_points()
    assert w.associate_partial(points)==('matched',1,[1])
    w.observe(geom(.2,.55,.17,fp=(.074,.074)),None)
    assert w.associate_partial(body_points(.2))==('matched',2,[2])
    w.objects[2].geometry=geom(.005,.55,.17,fp=(.074,.074))
    assert w.associate_partial(points)==('ambiguous',None,[1,2])


def test_merged_neighbours_insufficient_points_and_cap_slivers_do_not_bind():
    w=world();points=body_points()
    merged=np.vstack([points,body_points(.2)])
    assert w.associate_partial(merged)==('new',None,[])
    assert w.associate_partial(points[:10])==('new',None,[])
    cap=points.copy();cap[:,2]=np.linspace(.155,.168,len(cap))
    assert w.associate_partial(cap)==('new',None,[])
    assert w.associate_partial(np.full((40,3),np.nan))==('new',None,[])


@pytest.mark.parametrize('yaw',[0.,35.,90.])
def test_partial_volume_uses_stored_orientation(yaw):
    w=world();g=w.objects[1].geometry
    g.update(yaw_deg=yaw,extents=[.06,.04],footprint=[.06,.04])
    assert w.associate_partial(body_points(yaw=yaw))==('matched',1,[1])


def test_oblique_surface_normals_exclude_invalid_depth_and_their_neighbours():
    depth=np.ones((9,9));depth[4,4]=np.inf;depth[1,1]=np.nan;depth[7,7]=0.
    frame=NS(depth=depth,K=(100.,100.,4.,4.))
    with np.errstate(all='raise'):
        points=surface_points(frame,np.eye(4),stride=1)
    assert np.isfinite(points).all() and np.all(points[:,2]==1.)
    assert len(points)==81-3*5
    assert not np.any(np.linalg.norm(points[:,:2],axis=1)<.009)  # invalid centre isn't a synthetic surface


def test_clean_mask_projects_only_finite_depth_before_robot_exclusion():
    depth=np.array([[np.inf,.4,np.nan],[.4,0.,.4],[.4,7.,.4]])
    frame=NS(depth=depth,K=(100.,100.,1.,1.))
    seen=[]
    def contains(points):
        assert np.isfinite(points).all()
        seen.append(points)
        result=np.zeros(len(points),bool);result[0]=True
        return result
    with np.errstate(all='raise'):
        mask,fraction=clean_mask(np.ones_like(depth,bool),frame,np.eye(4),NS(contains=contains))
    assert len(seen[0])==5 and mask.sum()==4 and fraction==pytest.approx(.2)
    assert not mask[~np.isfinite(depth)].any()


def test_all_invalid_depth_yields_no_points_or_robot_queries():
    frame=NS(depth=np.full((9,9),np.inf),K=(100.,100.,4.,4.))
    with np.errstate(all='raise'):
        assert surface_points(frame,np.eye(4),stride=1).shape==(0,3)
        mask,fraction=clean_mask(np.ones_like(frame.depth,bool),frame,np.eye(4),
                                 NS(contains=lambda p:pytest.fail('no invalid points reach the robot filter')))
    assert not mask.any() and fraction==0.
