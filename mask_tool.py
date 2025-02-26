#!/usr/bin/env python
# @license
# Copyright 2018 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# %%
from __future__ import print_function, division
import time
import argparse
import collections
import copy
import json
import math
import os
import uuid
import webbrowser

import numpy as np
import six

import neuroglancer
neuroglancer.set_server_bind_address("0.0.0.0")
import neuroglancer.url_state
from neuroglancer.json_utils import json_encoder_default

def _full_count_for_level(level):
    return (2**level)**3

MAX_BLOCK_LEVEL = 5

class BlockMask(object):
    def __init__(self, max_level=MAX_BLOCK_LEVEL):
        # self.blocks[level_i][position] specifies the number of base elements contained within the block
        self.blocks = [dict() for _ in range(max_level+1)]

    def _remove_children(self, level, position):
        position = tuple(position)
        if level == 0:
            return
        for offset in np.ndindex((2,) * 3):
            self._remove(level - 1, tuple(x * 2 + o for x, o in zip(position, offset)))

    def _remove(self, level, position):
        position = tuple(position)
        old_count = self.blocks[level].pop(position, 0)
        if old_count == 0 or old_count == _full_count_for_level(level):
            return old_count
        self._remove_children(level, position)
        return old_count

    def _contains(self, level, position):
        position = tuple(position)
        blocks = self.blocks
        while True:
            count = blocks[level].get(position, 0)
            if count == _full_count_for_level(level):
                return level, position
            position = tuple(x // 2 for x in position)
            level += 1
            if level >= len(blocks):
                return None, None

    def _add_children(self, level, position, excluded_child_position, excluded_child_count):
        blocks = self.blocks
        full_count_for_child = _full_count_for_level(level-1)
        for offset in np.ndindex((2,) * 3):
            child_position = tuple(x * 2 + o for x, o in zip(position, offset))
            count = full_count_for_child
            if child_position == excluded_child_position:
                count -= excluded_child_count
            if count != 0:
                blocks[level-1][child_position] = count

    def _add_children_along_path(self, start_level, end_level, start_position):
        excluded_count = _full_count_for_level(start_level)
        while start_level < end_level:
            parent_position = tuple(x // 2 for x in start_position)
            start_level += 1
            self._add_children(start_level, parent_position, start_position, excluded_count)
            start_position = parent_position


    def add(self, level, position):
        if self._contains(level, position)[0] is not None:
            return
        position = tuple(position)
        old_count = self.blocks[level].get(position, 0)
        self._adjust_count(level, position, _full_count_for_level(level) - old_count)

    def add_or_remove_sphere(self, position, radius, add=True):
        radius = np.array(radius, dtype=np.int64)
        for off in np.ndindex(tuple(radius * 2 + 1)):
            off = off - radius
            if sum((off / radius)**2) >= 1: continue
            if add:
                self.add(0, position + off)
            else:
                self.remove(0, position + off)

    def remove(self, level, position):
        position = tuple(position)
        old_count = self.blocks[level].get(position, 0)
        if old_count == 0:
            old_level, position_in_old_level = self._contains(level, position)
            if old_level is None:
                return
            if old_level != level:
                self._adjust_count(old_level, position_in_old_level, -_full_count_for_level(level))
                self._add_children_along_path(level, old_level, position)
                return
        if old_count != _full_count_for_level(level):
            self._remove_children(level, position)
        self._adjust_count(level, position, -old_count)

    def _adjust_count(self, level, position, amount):
        if amount == 0:
            return
        old_count = self.blocks[level].get(position, 0)
        new_count = old_count + amount
        if new_count == 0:
            del self.blocks[level][position]
        else:
            self.blocks[level][position] = new_count

        if level > 0 and new_count == _full_count_for_level(level):
            self._remove_children(level, position)

        if level + 1 < len(self.blocks):
            self._adjust_count(level + 1, tuple(x // 2 for x in position), amount)

def make_block_mask(annotations, block_size, max_level=MAX_BLOCK_LEVEL):
    mask = BlockMask(max_level=max_level)
    for x in annotations:
        if not isinstance(x, neuroglancer.AxisAlignedBoundingBoxAnnotation):
            print('Warning: got non-box annotation: %r' % (x,))
            continue
        size = (x.point_b - x.point_a) / block_size
        if size[0] != int(size[0]) or np.any(size != size[0]):
            print('Warning: got invalid box: %r' % (x,))
            continue
        level = math.log(size[0]) / math.log(2)
        if level != int(level):
            print('Warning: got invalid box: %r' % (x,))
            continue
        level = int(level)
        eff_block_size = block_size * (2**level)
        if np.any(x.point_a % eff_block_size != 0):
            print('Warning: got invalid box: %r' % (x,))
            continue
        position = tuple(int(z) for z in x.point_a // eff_block_size)
        mask.add(level, position)
    return mask

def make_annotations_from_mask(mask, block_size):
    result = []
    for level, position_counts in enumerate(mask.blocks):
        full_count = _full_count_for_level(level)
        eff_block_size = block_size * 2**level
        for position in position_counts:
            count = position_counts[position]
            if count != full_count:
                continue
            position = np.array(position, dtype=np.int64)
            box_start = eff_block_size * position
            box_end = box_start + eff_block_size
            result.append(neuroglancer.AxisAlignedBoundingBoxAnnotation(
                point_a = box_start,
                point_b = box_end,
                id = uuid.uuid4().hex,
            ))
    return result


def normalize_block_annotations(annotations, block_size, max_level=3):
    mask = make_block_mask(annotations=annotations, block_size=block_size, max_level=max_level)
    return make_annotations_from_mask(mask=mask, block_size=block_size)


class Annotator(object):
    def __init__(self):
        self.annotation_layer_name = 'false-merges'
        self.false_merge_block_size = np.array([1, 1, 1], dtype=np.int64)
        self.cur_block_level = 2
        self.max_block_levels = 5
        viewer = self.viewer = neuroglancer.Viewer()
        self.other_state_segment_ids = dict()

        viewer.actions.add('anno-save', lambda s: self.save())
        viewer.actions.add('anno-mark-pre', lambda s: self.mark_synapse(s, layer='pre', add=True))
        viewer.actions.add('anno-mark-pre-sphere', lambda s: self.mark_synapse_sphere(s, layer='pre', add=True))
        viewer.actions.add('anno-unmark-pre', lambda s: self.unmark_synapse(s, layer='pre', add=False))
        viewer.actions.add('anno-mark-post', lambda s: self.mark_synapse(s, layer='post', add=True))
        viewer.actions.add('anno-mark-post-sphere', lambda s: self.mark_synapse_sphere(s, layer='post', add=True))
        viewer.actions.add('anno-unmark-post', lambda s: self.unmark_synapse(s, layer='post', add=False))
        viewer.actions.add('anno-decrease-block-size', self.decrease_block_size)
        viewer.actions.add('anno-increase-block-size', self.increase_block_size)

        with viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['bracketleft'] = 'anno-decrease-block-size'
            s.input_event_bindings.data_view['bracketright'] = 'anno-increase-block-size'
            s.input_event_bindings.data_view['control+keys'] = 'anno-save'
            s.input_event_bindings.data_view['keyt'] = 'anno-mark-pre'
            s.input_event_bindings.data_view['keyj'] = 'anno-mark-pre-sphere'
            s.input_event_bindings.data_view['control+shift+mousedown0'] = 'anno-unmark-pre'
            s.input_event_bindings.data_view['control+mousedown2'] = 'anno-mark-post'
            s.input_event_bindings.viewer['keyk'] = 'anno-mark-post-sphere'

            s.input_event_bindings.data_view['control+shift+mousedown2'] = 'anno-unmark-post'

        self.cur_message = None

    def increase_block_size(self, s):
        self.cur_block_level = min(self.max_block_levels, self.cur_block_level + 1)
        self.update_message()

    def decrease_block_size(self, s):
        self.cur_block_level = max(0, self.cur_block_level - 1)
        self.update_message()

    def mark_synapse(self, s, layer, add):
        voxel_coordinates = s.mouse_voxel_coordinates
        if voxel_coordinates is None:
            return
        block_size = self.false_merge_block_size
        level = self.cur_block_level
        with self.viewer.txn() as s:
            if s.layers.index(layer) == -1:
                s.layers[layer] = neuroglancer.LocalAnnotationLayer(
                    dimensions=s.dimensions,
                    shader='''
void main() {
  setBoundingBoxBorderWidth(0.0);
  setBoundingBoxFillColor(defaultColor());
}
''',
                    annotation_color = '#0f0' if layer == 'pre' else '#00f',
                )
            annotations = s.layers[layer].annotations
            t1  = time.time()
            mask = make_block_mask(annotations=annotations, block_size=block_size, max_level=self.max_block_levels)
            t2 = time.time()
            print("Time to make mask: ", t2 - t1)
            t3 = time.time()
            mask.add_or_remove_sphere(np.array([int(x) for x in voxel_coordinates]),
                                      np.array([1, 1, 1]) * 2**level,
                                      add=add)
            t4 = time.time()
            print("Time to add/remove sphere: ", t4 - t3)
            new_annotations = make_annotations_from_mask(mask=mask, block_size=block_size)
            t5 = time.time()
            print("Time to make annotations from mask: ", t5 - t4)
            print(level, new_annotations)
            s.layers[layer].annotations = new_annotations
            print("asdfadf")

    def mark_synapse_sphere(self, s, layer, add):
        voxel_coordinates = s.mouse_voxel_coordinates
        print(voxel_coordinates)
        if voxel_coordinates is None:
            return
        block_size = self.false_merge_block_size
        level = self.cur_block_level
        with self.viewer.txn() as s:
            if s.layers.index(layer) == -1:
                s.layers[layer] = neuroglancer.LocalAnnotationLayer(
                    dimensions=s.dimensions,
                    shader='''
void main() {
  setBoundingBoxBorderWidth(0.0);
  setBoundingBoxFillColor(defaultColor());
  setEllipsoidFillColor(defaultColor());
}
''',
                    annotation_color = '#0f0' if layer == 'pre' else '#00f',
                )
            annotations = list(s.layers[layer].annotations)
            t1  = time.time()
            annotations.append(neuroglancer.EllipsoidAnnotation(
                center=voxel_coordinates,
                radii=[2**level, 2**level, 2**level],
                id = uuid.uuid4().hex,
            ))
            s.layers[layer].annotations = annotations

    def update_message(self):
        message = '[Block size: %d vx] ' % (self.false_merge_block_size[0] )#* 2**self.false_merge_block_level)
        if message != self.cur_message:
            with self.viewer.config_state.txn() as s:
                if message is not None:
                    s.status_messages['status'] = message
                else:
                    s.status_messages.pop('status')
            self.cur_message = message

    def show(self):
        webbrowser.open_new(self.viewer.get_viewer_url())

    def get_viewer_url(self):
        return self.viewer.get_viewer_url()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', type=str)
    ap.add_argument(
        '-n', '--no-webbrowser', action='store_true', help='Don\'t open the webbrowser.')
    ap.add_argument(
        '-a',
        '--bind-address',
        help='Bind address for Python web server.  Use 127.0.0.1 (the default) to restrict access '
        'to browers running on the local machine, use 0.0.0.0 to permit access from remote browsers.'
    )
    ap.add_argument(
        '--static-content-url', help='Obtain the Neuroglancer client code from the specified URL.')

    args = ap.parse_args()
    if args.bind_address:
        neuroglancer.set_server_bind_address(args.bind_address)
    if args.static_content_url:
        neuroglancer.set_static_content_source(url=args.static_content_url)

    anno = Annotator()

    if args.url:
        anno.viewer.set_state(neuroglancer.parse_url(args.url))

    if not args.no_webbrowser:
        anno.show()
    while True:
        pass

# %%
# import urllib

# res = urllib.parse.urlparse('https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22z%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22x%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B24822%2C10725%2C11800%5D%2C%22crossSectionScale%22:1%2C%22projectionScale%22:65536%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/%7Czarr2:%22%2C%22tab%22:%22source%22%2C%22name%22:%22fibsem-uint8%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22fibsem-uint8%22%7D%2C%22layout%22:%224panel%22%7D')
# json_string = neuroglancer.url_state.url_fragment_to_json(res.fragment)

# # %%
# json_string
# # %%
# anno = Annotator()
# url = 'https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B19998.5%2C3998.5%2C18624.5%5D%2C%22crossSectionOrientation%22:%5B1%2C0%2C0%2C0%5D%2C%22crossSectionScale%22:50%2C%22projectionOrientation%22:%5B1%2C0%2C0%2C0%5D%2C%22projectionScale%22:65536%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22zarr://s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.zarr/recon-1/em/fibsem-uint8/%22%2C%22tab%22:%22rendering%22%2C%22opacity%22:1%2C%22blend%22:%22additive%22%2C%22shader%22:%22#uicontrol%20invlerp%20normalized%28range=%5B203%2C%20238%5D%2C%20window=%5B0%2C%20255%5D%29%5Cn#uicontrol%20vec3%20color%20color%28default=%5C%22white%5C%22%29%5Cnvoid%20main%28%29%7BemitRGB%28color%20%2A%20normalized%28%29%29%3B%7D%22%2C%22name%22:%22fibsem-uint16%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22fibsem-uint16%22%7D%2C%22crossSectionBackgroundColor%22:%22#000000%22%2C%22layout%22:%224panel%22%7D'
# anno.viewer.set_state(neuroglancer.parse_url(url))
# # layer = "pre"
# # with anno.viewer.txn() as s:
# #     if s.layers.index(layer) == -1:
# #         s.layers[layer] = neuroglancer.LocalAnnotationLayer(
# #             dimensions=s.dimensions,
# #             shader='''
# # void main() {
# # setBoundingBoxBorderWidth(0.0);
# # setBoundingBoxFillColor(defaultColor());
# # }
# # ''',
# #             annotation_color = '#0f0' if layer == 'pre' else '#00f',
# #         )
# anno.show()

# # %%
# import neuroglancer
# import uuid
# print(neuroglancer.EllipsoidAnnotation(center=[0, 0, 0], radii=[1, 1, 1], id = uuid.uuid4().hex))
# print(neuroglancer.AxisAlignedBoundingBoxAnnotation(point_a=[0, 0, 0], point_b=[1, 1, 1]))
# %%
