<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data processing

## Find the farthest camera pair

Since we want to divide all the cameras into two groups and subsample cameras from one of the group, we first use the function **find_farthest_camera_pair** to get the indices of the farthest camera pair. These two cameras are treated as the reference to assign other cameras to which group.

## Divide all the cameras into two groups

Use the first camera in the farthest camera pair as the reference, and use **rank_views_by_pose_distance** to rank all the cameras w.r.t the distance between the reference camera and all cameras, then equally divide them into two groups. **valid_mask_first_half** denotes the indices of the cameras in the first group, and **valid_mask_second_half** denotes the indices of the cameras in the second group.

## Sparsely sample cameras in the group

After we divide the cameras into two groups, we want to select cameras in one group to do sparse reconstruction. The function **similarity_sampling** takes initial camera idx, extrinsics, and valid_mask as inputs. We use an array **sampled_indices** to save the sampled camera indices, starting from the initial camera. We use a list called **min_distances** to record the minimum distance between the i'th camera and the cameras that have been visited. Each iteration, we select the camera in **min_distances** that has the largest distance between it and all the cameras in **sampled_indices**. Then put the new selected camera idx into **sampled_indices** and update the **min_distances** list for the next round. In the end, **similarity_sampling** return the array **sampled_indices**.
