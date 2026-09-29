Reviewing instances
================================
The dashboard's **Review** tab walks the instances of a segmentation one at a time and records a verdict on each: *blessed* (correct), *edited* (partly correct, with notes) or *erased* (a false positive). It moves the neuroglancer viewer to each instance as you go. The verdicts are kept in a SQLite file, the review index, which you build once per segmentation.

Build an index
--------------

.. code-block:: bash

    python -m cellmap_flow.review_index /path/to/instances.zarr/labels review.sqlite

The segmentation is anything cellmap-flow reads (zarr v2 or v3, N5, precomputed), with 0 as background and axes z, y, x. A multiscale group is read at its finest level. The volume is read in blocks of whole chunks, so it does not have to fit in memory.

For each label the index stores its voxel count, its bounding box in voxels, and its centroid, both in voxels and in world nanometres. The centroid is what the tab navigates to. The file also has an empty ledger row per label and some metadata: the source path, the voxel size, and the lower corner of voxel 0.

Options:

- ``--queue {smallest,largest,random}``: an order to review in, repeatable. The default is ``smallest`` and ``random``.
- ``--seed``: the seed of the ``random`` queue.
- ``--block Z Y X``: how many voxels to read at a time.

The builder never overwrites an existing index, because its ledger holds the verdicts. To rebuild one, move the old file away first.

Queues are data, not names built into the dashboard. Every ``rank_<name>`` column of the ``instances`` table is a queue called ``<name>``, reviewed in ascending rank; rows where it is NULL are left out. You can add your own ordering to an index, for example by mean intensity or by a false-merge score, with ``ALTER TABLE instances ADD COLUMN rank_<name> INTEGER`` and an ``UPDATE``. The tab will offer it the next time it refreshes. A ``queue_labels`` entry in the ``meta`` table, a JSON object from queue name to a description, labels the queues in the tab.

Review in the dashboard
-----------------------

1. Open the **Review** tab and enter the index's path. Only files named ``.sqlite`` or ``.db`` are opened.
2. Optionally, enter a reviewer name and the name of the viewer's segmentation layer. With the layer set, you can hover over a segment in neuroglancer and press ``t`` to pick that instance.
3. Choose a queue and, optionally, a minimum voxel count. Then press **Next**. The viewer moves to the next unreviewed instance.
4. Press **Bless**, **Edit** or **Erase**. With *Auto-advance* ticked, the tab then moves on to the next instance. **Undo** clears the verdict on the instance shown, and **Go to ID** jumps to a given label.

The progress bar counts the verdicts per state and per queue. The dashboard opens the index read-only, except to record or undo a verdict.
