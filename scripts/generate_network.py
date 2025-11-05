#!/usr/bin/env python3
"""
SUMO Network Generation Script

Generates a simple 1x1 intersection network for traffic light control.
Creates the necessary .net.xml, .rou.xml, and .sumo.cfg files.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from traffic_rl.utils import check_sumo_installation


def generate_network_xml():
    """Generate the network XML file for a 1x1 intersection."""
    network_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.5">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <!-- Edges -->
    <edge id="north" from="north_junction" to="center" priority="1">
        <lane id="north_0" index="0" speed="13.89" length="100.00" shape="100.00,0.00 0.00,0.00"/>
    </edge>
    
    <edge id="south" from="center" to="south_junction" priority="1">
        <lane id="south_0" index="0" speed="13.89" length="100.00" shape="0.00,100.00 100.00,100.00"/>
    </edge>
    
    <edge id="east" from="center" to="east_junction" priority="1">
        <lane id="east_0" index="0" speed="13.89" length="100.00" shape="100.00,0.00 100.00,100.00"/>
    </edge>
    
    <edge id="west" from="west_junction" to="center" priority="1">
        <lane id="west_0" index="0" speed="13.89" length="100.00" shape="0.00,100.00 0.00,0.00"/>
    </edge>
    
    <!-- Junctions -->
    <junction id="north_junction" type="priority" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00 0.00,-10.00 -10.00,-10.00 -10.00,0.00 0.00,0.00"/>
    <junction id="south_junction" type="priority" x="100.00" y="100.00" incLanes="" intLanes="" shape="100.00,100.00 100.00,110.00 110.00,110.00 110.00,100.00 100.00,100.00"/>
    <junction id="east_junction" type="priority" x="100.00" y="0.00" incLanes="" intLanes="" shape="100.00,0.00 110.00,0.00 110.00,10.00 100.00,10.00 100.00,0.00"/>
    <junction id="west_junction" type="priority" x="0.00" y="100.00" incLanes="" intLanes="" shape="0.00,100.00 -10.00,100.00 -10.00,90.00 0.00,90.00 0.00,100.00"/>
    
    <!-- Center junction with traffic light -->
    <junction id="center" type="traffic_light" x="50.00" y="50.00" incLanes="north_0 west_0" intLanes="center_0 center_1" shape="50.00,50.00 50.00,40.00 40.00,40.00 40.00,50.00 50.00,50.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>
    
    <!-- Internal lanes -->
    <junction id="center_internal" type="internal" x="50.00" y="50.00" incLanes="center_0 center_1" intLanes="south_0 east_0" shape="50.00,50.00 50.00,60.00 60.00,60.00 60.00,50.00 50.00,50.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
    </junction>
    
    <!-- Internal edges -->
    <edge id="center" from="center" to="center_internal" priority="1">
        <lane id="center_0" index="0" speed="13.89" length="10.00" shape="50.00,50.00 50.00,60.00"/>
        <lane id="center_1" index="1" speed="13.89" length="10.00" shape="40.00,50.00 60.00,50.00"/>
    </edge>
    
    <!-- Connections -->
    <connection from="north" to="center" fromLane="0" toLane="0" via="center_0" dir="s" state="M"/>
    <connection from="west" to="center" fromLane="0" toLane="1" via="center_1" dir="l" state="M"/>
    <connection from="center" to="south" fromLane="0" toLane="0" via="south_0" dir="s" state="M"/>
    <connection from="center" to="east" fromLane="1" toLane="0" via="east_0" dir="r" state="M"/>
</net>'''
    
    network_path = project_root / "network" / "intersection.net.xml"
    network_path.parent.mkdir(exist_ok=True)
    
    with open(network_path, 'w') as f:
        f.write(network_content)
    
    print(f"✓ Network file created: {network_path}")
    return network_path


def generate_routes_xml():
    """Generate the routes XML file with vehicle flows."""
    routes_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
    
    <!-- Routes -->
    <route id="north_to_south" edges="north center south"/>
    <route id="south_to_north" edges="south center north"/>
    <route id="west_to_east" edges="west center east"/>
    <route id="east_to_west" edges="east center west"/>
    
    <!-- Vehicle flows -->
    <flow id="north_flow" type="car" route="north_to_south" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="south_flow" type="car" route="south_to_north" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="west_flow" type="car" route="west_to_east" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="east_flow" type="car" route="east_to_west" begin="0" end="3600" vehsPerHour="300"/>
</routes>'''
    
    routes_path = project_root / "routes" / "intersection.rou.xml"
    routes_path.parent.mkdir(exist_ok=True)
    
    with open(routes_path, 'w') as f:
        f.write(routes_content)
    
    print(f"✓ Routes file created: {routes_path}")
    return routes_path


def generate_sumo_config():
    """Generate the main SUMO configuration file."""
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="network/intersection.net.xml"/>
        <route-files value="routes/intersection.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <ignore-junction-blocker value="1"/>
    </processing>
    
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>'''
    
    config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ SUMO config file created: {config_path}")
    return config_path


def main():
    """Main function to generate all SUMO files."""
    print("🚦 Generating SUMO network files...")
    
    # Check SUMO installation (optional for mock mode)
    if not check_sumo_installation():
        print("⚠️  SUMO installation not found. Generating files for mock mode.")
    
    # Generate files
    network_path = generate_network_xml()
    routes_path = generate_routes_xml()
    config_path = generate_sumo_config()
    
    print("\n✅ All SUMO files generated successfully!")
    print(f"   Network: {network_path}")
    print(f"   Routes:  {routes_path}")
    print(f"   Config:  {config_path}")
    print("\nYou can now run training with: python scripts/train.py")


if __name__ == "__main__":
    main()
