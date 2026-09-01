Matterport3D dataset prepared for use with Habitat
--------------------------------------------------

This archive contains the [Matterport3D](https://niessner.github.io/Matterport/) dataset prepared for use with the [Habitat](https://aihabitat.org) platform.
The `*.glb` files are glTF v2 binary format versions of the original Matterport3D `.obj` mesh files.
The `*.house` files are identical to the ones in the Matterport3D `house_segmentations` archives.
The `*_semantic.ply` files are PLY binary format meshes containing semantic instance information (created from the original `house_segmentations` PLY files using the Habitat datatool).

Changelog
---------

- 2020-02-04: Updated `*.glb` files to use basis compression for textures
- 2019-04-01: Fixed bug in `*_semantic.ply` files resulting in erroneous object ids (caused errors in semantic mask sensor frame data)

