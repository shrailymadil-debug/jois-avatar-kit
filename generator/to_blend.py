import bpy, sys, os
src, out_dir = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
# keep names clean in the Blender scene
for o in bpy.data.objects:
    if o.type == "ARMATURE": o.name = "Armature"; o.data.name = "Armature"
for o in list(bpy.data.objects):
    if o.type == "MESH" and o.name != "Jois_Avatar": bpy.data.objects.remove(o)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "jois_avatar.blend"), compress=True)
bpy.ops.export_scene.fbx(filepath=os.path.join(out_dir, "jois_avatar.fbx"), path_mode="COPY", embed_textures=True,
                         add_leaf_bones=False, use_mesh_modifiers=False, bake_anim=False, mesh_smooth_type="FACE")
me = bpy.data.objects["Jois_Avatar"]
print("MESH", me.name, "shape keys:", len(me.data.shape_keys.key_blocks) - 1, "verts:", len(me.data.vertices),
      "tris:", sum(len(p.vertices) - 2 for p in me.data.polygons), "bones:", len([o for o in bpy.data.objects if o.type == "ARMATURE"][0].data.bones))
