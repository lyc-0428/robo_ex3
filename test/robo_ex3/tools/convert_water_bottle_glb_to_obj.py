import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GLB = ROOT / "models" / "water_bottle" / "meshes" / "WaterBottle.glb"
OBJ = ROOT / "models" / "water_bottle" / "meshes" / "WaterBottle_fortress.obj"
MTL = ROOT / "models" / "water_bottle" / "meshes" / "WaterBottle_fortress.mtl"


def read_glb(path):
    data = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total_length != len(data):
        raise ValueError("Expected a valid GLB 2.0 file")

    offset = 12
    json_length, json_type = struct.unpack_from("<II", data, offset)
    offset += 8
    if json_type != 0x4E4F534A:
        raise ValueError("The first GLB chunk is not JSON")
    document = json.loads(data[offset : offset + json_length].decode("utf-8"))
    offset += json_length

    bin_length, bin_type = struct.unpack_from("<II", data, offset)
    offset += 8
    if bin_type != 0x004E4942:
        raise ValueError("The second GLB chunk is not BIN")
    return document, memoryview(data)[offset : offset + bin_length]


def read_accessor(document, binary, accessor_index):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_format = {5123: "H", 5125: "I", 5126: "f"}[accessor["componentType"]]
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    element_format = "<" + component_format * component_count
    element_size = struct.calcsize(element_format)
    stride = view.get("byteStride", element_size)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return [
        struct.unpack_from(element_format, binary, start + index * stride)
        for index in range(accessor["count"])
    ]


def to_z_up(vector):
    # Apply the GLB node's 180-degree Y rotation, then convert Y-up to Z-up.
    x, y, z = vector
    return -x, z, y


def main():
    document, binary = read_glb(GLB)
    primitive = document["meshes"][0]["primitives"][0]
    positions = read_accessor(document, binary, primitive["attributes"]["POSITION"])
    normals = read_accessor(document, binary, primitive["attributes"]["NORMAL"])
    texcoords = read_accessor(document, binary, primitive["attributes"]["TEXCOORD_0"])
    indices = read_accessor(document, binary, primitive["indices"])

    lines = ["mtllib WaterBottle_fortress.mtl", "o WaterBottle", "usemtl BottleMat"]
    lines.extend("v {:.9g} {:.9g} {:.9g}".format(*to_z_up(value)) for value in positions)
    lines.extend("vt {:.9g} {:.9g}".format(value[0], 1.0 - value[1]) for value in texcoords)
    lines.extend("vn {:.9g} {:.9g} {:.9g}".format(*to_z_up(value)) for value in normals)
    for start in range(0, len(indices), 3):
        triangle = [indices[start + offset][0] + 1 for offset in range(3)]
        lines.append("f " + " ".join(f"{index}/{index}/{index}" for index in triangle))
    OBJ.write_text("\n".join(lines) + "\n", encoding="utf-8")

    MTL.write_text(
        "\n".join(
            [
                "newmtl BottleMat",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ks 0.15 0.15 0.15",
                "Ns 32.0",
                "d 1.0",
                "illum 2",
                "map_Kd ../materials/textures/WaterBottle_BottleMat_Diffuse.png",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
