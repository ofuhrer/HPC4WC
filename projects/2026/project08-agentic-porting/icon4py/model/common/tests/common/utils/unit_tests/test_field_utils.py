# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import field_utils


@pytest.fixture
def allocator():
    return np


class TestFlip:
    def test_flip_1d(self, allocator):
        field = gtx.as_field({dims.CellDim: range(0, 4)}, np.array([10.0, 20.0, 30.0, 40.0]))
        result = field_utils.flip(field, dims.CellDim, allocator=allocator)

        assert np.array_equal(result.ndarray, [40.0, 30.0, 20.0, 10.0])
        assert result.domain == field.domain

    def test_flip_1d_nonzero_start(self, allocator):
        field = gtx.as_field({dims.CellDim: range(3, 7)}, np.array([10.0, 20.0, 30.0, 40.0]))
        result = field_utils.flip(field, dims.CellDim, allocator=allocator)

        assert np.array_equal(result.ndarray, [40.0, 30.0, 20.0, 10.0])
        assert result.domain == field.domain

    def test_flip_2d_along_first_dim(self, allocator):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        field = gtx.as_field({dims.CellDim: range(0, 2), dims.KDim: range(0, 3)}, data)
        result = field_utils.flip(field, dims.CellDim, allocator=allocator)

        expected = np.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == field.domain

    def test_flip_2d_along_second_dim(self, allocator):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        field = gtx.as_field({dims.CellDim: range(0, 2), dims.KDim: range(0, 3)}, data)
        result = field_utils.flip(field, dims.KDim, allocator=allocator)

        expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == field.domain

    def test_flip_2d_nonzero_start_along_first_dim(self, allocator):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        field = gtx.as_field({dims.CellDim: range(5, 7), dims.KDim: range(2, 5)}, data)
        result = field_utils.flip(field, dims.CellDim, allocator=allocator)

        expected = np.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == field.domain

    def test_flip_2d_nonzero_start_along_second_dim(self, allocator):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        field = gtx.as_field({dims.CellDim: range(5, 7), dims.KDim: range(2, 5)}, data)
        result = field_utils.flip(field, dims.KDim, allocator=allocator)

        expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == field.domain

    def test_flip_preserves_dtype(self, allocator):
        field = gtx.as_field({dims.KDim: range(3, 6)}, np.array([1, 2, 3], dtype=np.int32))
        result = field_utils.flip(field, dims.KDim, allocator=allocator)

        assert result.ndarray.dtype == np.int32
        assert np.array_equal(result.ndarray, [3, 2, 1])


class TestIndex2Offset:
    def test_1d_zero_start(self, allocator):
        # indices: [2, 1, 3, 0], positions: [0, 1, 2, 3]
        # offsets: [2-0, 1-1, 3-2, 0-3] = [2, 0, 1, -3]  # noqa: ERA001
        index_field = gtx.as_field(
            {dims.CellDim: range(0, 4)}, np.array([2, 1, 3, 0], dtype=np.int32)
        )
        result = field_utils.index2offset(index_field, dims.CellDim, allocator=allocator)

        assert np.array_equal(result.ndarray, [2, 0, 1, -3])
        assert result.domain == index_field.domain

    def test_1d_nonzero_start(self, allocator):
        # indices: [5, 3, 4, 2], positions: [2, 3, 4, 5]
        # offsets: [5-2, 3-3, 4-4, 2-5] = [3, 0, 0, -3]  # noqa: ERA001
        index_field = gtx.as_field(
            {dims.CellDim: range(2, 6)}, np.array([5, 3, 4, 2], dtype=np.int32)
        )
        result = field_utils.index2offset(index_field, dims.CellDim, allocator=allocator)

        assert np.array_equal(result.ndarray, [3, 0, 0, -3])
        assert result.domain == index_field.domain

    def test_1d_identity_permutation(self, allocator):
        # indices == positions => all offsets are 0
        index_field = gtx.as_field({dims.KDim: range(3, 7)}, np.array([3, 4, 5, 6], dtype=np.int32))
        result = field_utils.index2offset(index_field, dims.KDim, allocator=allocator)

        assert np.array_equal(result.ndarray, [0, 0, 0, 0])

    def test_2d_along_second_dim(self, allocator):
        # shape (3, 4), apply along KDim (axis=1)
        # positions for KDim with range(0, 4): [0, 1, 2, 3]
        data = np.array([[0, 3, 1, 2], [1, 0, 2, 3], [2, 1, 3, 0]], dtype=np.int32)
        index_field = gtx.as_field({dims.CellDim: range(0, 3), dims.KDim: range(0, 4)}, data)
        result = field_utils.index2offset(index_field, dims.KDim, allocator=allocator)

        expected = np.array([[0, 2, -1, -1], [1, -1, 0, 0], [2, 0, 1, -3]], dtype=np.int32)
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == index_field.domain

    def test_2d_nonzero_start_along_second_dim(self, allocator):
        # shape (2, 3), KDim starts at 5
        # positions for KDim with range(5, 8): [5, 6, 7]
        data = np.array([[7, 5, 6], [6, 7, 5]], dtype=np.int32)
        index_field = gtx.as_field({dims.CellDim: range(2, 4), dims.KDim: range(5, 8)}, data)
        result = field_utils.index2offset(index_field, dims.KDim, allocator=allocator)

        # offsets: [[7-5, 5-6, 6-7], [6-5, 7-6, 5-7]] = [[2, -1, -1], [1, 1, -2]]  # noqa: ERA001
        expected = np.array([[2, -1, -1], [1, 1, -2]], dtype=np.int32)
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == index_field.domain

    def test_2d_along_first_dim_nonzero_start(self, allocator):
        # shape (3, 2), apply along CellDim (axis=0)
        # positions for CellDim with range(10, 13): [10, 11, 12]
        # arange is 1D along CellDim axis, broadcast subtracted from (3, 2) data
        data = np.array([[12, 11], [10, 12], [11, 10]], dtype=np.int32)
        index_field = gtx.as_field({dims.CellDim: range(10, 13), dims.KDim: range(0, 2)}, data)
        result = field_utils.index2offset(index_field, dims.CellDim, allocator=allocator)

        # offsets: [[12-10, 11-10], [10-11, 12-11], [11-12, 10-12]] = [[2, 1], [-1, 1], [-1, -2]]  # noqa: ERA001
        expected = np.array([[2, 1], [-1, 1], [-1, -2]], dtype=np.int32)
        assert np.array_equal(result.ndarray, expected)
        assert result.domain == index_field.domain
