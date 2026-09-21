#!/usr/bin/env python3

"""Guards on the package manifest and the waypoints file.

A malformed package.xml fails at `rosdep install` time, before anything is
built, with a parse error rather than an obvious message -- worth catching
here instead. The waypoints check exists because the goal list is graded:
a typo in it silently costs points with no error anywhere.

Standard library plus PyYAML only, so this runs anywhere the nodes do.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_XML = PKG_ROOT / 'package.xml'
WAYPOINTS = PKG_ROOT / 'config' / 'waypoints.yaml'

# The official Town01 goal points for the 2025 competition.
OFFICIAL_GOALS = {
    (334.949799, -161.106171), (339.100037, -258.568939),
    (396.295319, -183.195740), (267.657074, -1.983160),
    (153.868896, -26.115866), (290.515564, -56.175072),
    (92.325722, -86.063644), (88.384346, -287.468567),
    (177.594101, -326.386902), (-1.646942, -197.501282),
    (59.701321, -1.970804), (122.100121, -55.142044),
    (161.030975, -129.313187), (184.758713, -199.424271),
}


def test_package_xml_is_well_formed():
    ET.parse(PACKAGE_XML)


def test_no_double_hyphen_inside_xml_comments():
    """'--' is illegal inside an XML comment and breaks the whole manifest."""
    text = PACKAGE_XML.read_text()
    for comment in re.findall(r'<!--.*?-->', text, re.S):
        assert '--' not in comment[4:-3], f"'--' inside comment: {comment[:60]!r}"


def test_package_name_matches_competition_requirement():
    """Submissions must provide a package named exactly shell_simulation."""
    assert ET.parse(PACKAGE_XML).getroot().findtext('name') == 'shell_simulation'


def test_manifest_declares_no_third_party_python_deps():
    """The run environment installs nothing, so these must not come back.

    Dependencies are resolved at build time against a different container
    than the one the nodes run in, so a third-party import that is declared
    here but missing at run time kills the node at start-up. That is exactly
    how networkx stopped the vehicle from ever moving.
    """
    root = ET.parse(PACKAGE_XML).getroot()
    declared = {e.text for e in root.iter() if e.tag.endswith('depend')}
    for banned in ('python3-networkx', 'networkx'):
        assert banned not in declared


def test_launch_file_exists_with_required_name():
    """The competition launches this exact path as the entry point."""
    assert (PKG_ROOT / 'launch' / 'shell_simulation.launch.py').is_file()


def test_waypoints_file_parses():
    data = yaml.safe_load(WAYPOINTS.read_text())
    assert 'ordered_waypoints' in data
    assert isinstance(data['ordered_waypoints'], list)


def test_waypoints_are_exactly_the_official_goals():
    data = yaml.safe_load(WAYPOINTS.read_text())['ordered_waypoints']
    got = {(round(p[0], 6), round(p[1], 6)) for p in data}
    expected = {(round(x, 6), round(y, 6)) for x, y in OFFICIAL_GOALS}
    assert got == expected, (
        f"missing: {expected - got}\nunexpected: {got - expected}")


def test_waypoints_have_no_duplicates():
    data = yaml.safe_load(WAYPOINTS.read_text())['ordered_waypoints']
    seen = {(round(p[0], 6), round(p[1], 6)) for p in data}
    assert len(seen) == len(data)


@pytest.mark.parametrize('index', range(14))
def test_each_waypoint_is_three_floats(index):
    """planning_node builds geometry_msgs/Point, which rejects ints."""
    point = yaml.safe_load(WAYPOINTS.read_text())['ordered_waypoints'][index]
    assert len(point) == 3
    assert all(isinstance(c, float) for c in point)


def test_spawn_point_is_not_listed_as_a_goal():
    """Routing from the spawn point to itself can hang the planner."""
    data = yaml.safe_load(WAYPOINTS.read_text())['ordered_waypoints']
    for p in data:
        assert not (abs(p[0] - 280.363739) < 0.5 and abs(p[1] + 129.306351) < 0.5)
