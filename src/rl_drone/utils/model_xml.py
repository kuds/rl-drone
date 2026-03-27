"""Utilities for modifying the MuJoCo CF2 drone model XML."""

import xml.etree.ElementTree as ET

from robot_descriptions import cf2_mj_description


def setup_mujoco_model(sphere_size: float = 0.25, target_height: float = 1.0):
    """Add fly_sensor site and touch_sensor to the CF2 MuJoCo model.

    Modifies the XML file in-place, adding elements only if they don't
    already exist.

    Args:
        sphere_size: Radius of the target sphere site.
        target_height: Z-coordinate of the target site.

    Returns:
        True if changes were written, False if all elements already existed.

    Raises:
        FileNotFoundError: If the CF2 MJCF file is not found.
        RuntimeError: If the XML is missing required <worldbody> or <sensor> nodes.
    """
    filename = cf2_mj_description.MJCF_PATH

    tree = ET.parse(filename)
    root = tree.getroot()

    worldbody_node = root.find("worldbody")
    sensor_node = root.find("sensor")

    if worldbody_node is None or sensor_node is None:
        missing = []
        if worldbody_node is None:
            missing.append("<worldbody>")
        if sensor_node is None:
            missing.append("<sensor>")
        raise RuntimeError(
            f"CF2 model XML is missing required nodes: {', '.join(missing)}"
        )

    changes_made = False

    if worldbody_node.find("./site[@name='fly_sensor']") is None:
        site_attributes = {
            "name": "fly_sensor",
            "pos": f"0 0 {target_height}",
            "size": str(sphere_size),
            "type": "sphere",
            "rgba": "0 1 0 .25",
        }
        ET.SubElement(worldbody_node, "site", attrib=site_attributes)
        changes_made = True

    if sensor_node.find("./touch[@name='touch_sensor']") is None:
        touch_attributes = {"name": "touch_sensor", "site": "fly_sensor"}
        ET.SubElement(sensor_node, "touch", attrib=touch_attributes)
        changes_made = True

    if changes_made:
        ET.indent(tree, space="  ", level=0)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    return changes_made
