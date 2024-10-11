"""Utils.py test module."""

import numpy as np
import pytest

from rainnow.src.data_processing import patch_2d_arr_into_nxn_squares
from rainnow.src.utilities.utils import calculate_required_1d_padding


def test_create_patches_from_2d_arr_using_overlap_square_input_arr_2x2_patch():
    """Test overlap patching function for input square arr."""
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    patch_size = 2  # 2x2
    x_pad = calculate_required_1d_padding(X=arr2d.shape[1], Y=patch_size, frac=1)
    y_pad = calculate_required_1d_padding(X=arr2d.shape[0], Y=patch_size, frac=1)

    desired_with_flip = np.array(
        [[[[7, 8], [4, 5]], [[8, 9], [5, 6]]], [[[4, 5], [1, 2]], [[5, 6], [2, 3]]]]
    )
    desired_without_flip = np.array(
        [[[[1, 2], [4, 5]], [[2, 3], [5, 6]]], [[[4, 5], [7, 8]], [[5, 6], [8, 9]]]]
    )

    actual_with_flip = patch_2d_arr_into_nxn_squares(
        arr2d=arr2d, n=patch_size, x_pad=x_pad, y_pad=y_pad, flip_pixels=True
    )
    actual_without_flip = patch_2d_arr_into_nxn_squares(
        arr2d=arr2d, n=patch_size, x_pad=x_pad, y_pad=y_pad, flip_pixels=False
    )

    assert np.all(desired_with_flip == actual_with_flip)
    assert np.all(desired_without_flip == actual_without_flip)


def test_create_patches_from_2d_arr_using_overlap_nonsquare_input_arr_2x2_patch():
    """Test overlap patching function for input square arr."""
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    patch_size = 2  # 2x2
    x_pad = calculate_required_1d_padding(X=arr2d.shape[1], Y=patch_size, frac=1)
    y_pad = calculate_required_1d_padding(X=arr2d.shape[0], Y=patch_size, frac=1)

    desired_with_flip = np.array(
        [
            [[[10, 11], [7, 8]], [[11, 12], [8, 9]]],
            [[[4, 5], [1, 2]], [[5, 6], [2, 3]]],
        ]
    )
    desired_without_flip = np.array(
        [
            [[[1, 2], [4, 5]], [[2, 3], [5, 6]]],
            [[[7, 8], [10, 11]], [[8, 9], [11, 12]]],
        ]
    )

    actual_with_flip = patch_2d_arr_into_nxn_squares(
        arr2d=arr2d, n=patch_size, x_pad=x_pad, y_pad=y_pad, flip_pixels=True
    )
    actual_without_flip = patch_2d_arr_into_nxn_squares(
        arr2d=arr2d, n=patch_size, x_pad=x_pad, y_pad=y_pad, flip_pixels=False
    )

    assert np.all(desired_with_flip == actual_with_flip)
    assert np.all(desired_without_flip == actual_without_flip)


def test_create_patches_from_2d_arr_using_overlap_arr_dims_too_small():
    """Test overlap patching function error when arr dims too small for patch_size."""
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    patch_size = 4
    x_pad = calculate_required_1d_padding(X=arr2d.shape[1], Y=patch_size, frac=1)
    y_pad = calculate_required_1d_padding(X=arr2d.shape[0], Y=patch_size, frac=1)

    with pytest.raises(AssertionError):
        _ = patch_2d_arr_into_nxn_squares(
            arr2d=arr2d,
            n=patch_size,
            x_pad=x_pad,
            y_pad=y_pad,
            flip_pixels=True,
        )


def test_create_patches_from_2d_arr_using_overlap_arr_not_2d():
    """Test overlap patching function error when input arr is not 2d."""
    arr3d = np.array([[[1, 2], [3, 4]]])
    patch_size = 1
    x_pad = calculate_required_1d_padding(X=arr3d.shape[0], Y=patch_size, frac=1)
    y_pad = calculate_required_1d_padding(X=arr3d.shape[1], Y=patch_size, frac=1)
    with pytest.raises(AssertionError):
        _ = patch_2d_arr_into_nxn_squares(
            arr2d=arr3d,
            n=patch_size,
            x_pad=x_pad,
            y_pad=y_pad,
            flip_pixels=True,
        )


def test_calculate_required_1d_padding_no_remainder():
    """Test typical values for calculate_required_1d_padding() when X/Y has no remainder."""
    # run the padding calc. using a range of frac values.
    actual = [
        calculate_required_1d_padding(X=2, Y=2, frac=-1),
        calculate_required_1d_padding(X=2, Y=2, frac=0),
        calculate_required_1d_padding(X=2, Y=2, frac=1),
        calculate_required_1d_padding(X=2, Y=2, frac=2),
        calculate_required_1d_padding(X=2, Y=2, frac=10),
    ]
    desired = [0, 0, 0, 0, 0]

    assert desired == actual


def test_calculate_required_1d_for_X_larger_Y():
    """Test typical values for calculate_required_1d_padding() when X/Y > 1 and has a remainder."""
    # for a/b and frac c. If b is a common factor of b then padding will always be 0.
    # For examples, 3 / 2 has remainder .5 therefore any even frac will return 0.
    # and any odd frac (other than 1 and 0) will return 1.
    actual = [
        calculate_required_1d_padding(X=3, Y=2, frac=-1),  # nearest int.
        calculate_required_1d_padding(X=3, Y=2, frac=0),  # nearest int.
        calculate_required_1d_padding(X=3, Y=2, frac=1),  # nearest int.
        calculate_required_1d_padding(X=3, Y=2, frac=2),  # nearest 1/2 int.
        calculate_required_1d_padding(X=3, Y=2, frac=2.1),
        calculate_required_1d_padding(X=3, Y=2, frac=3),
        calculate_required_1d_padding(X=3, Y=2, frac=20),
        calculate_required_1d_padding(X=10, Y=3, frac=1),  # 2 pad.
        calculate_required_1d_padding(X=10, Y=3, frac=2),
        calculate_required_1d_padding(X=10, Y=3, frac=5),
        calculate_required_1d_padding(X=10, Y=3, frac=99),
    ]

    desired = [1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 0]

    assert actual == desired


def test_calculate_required_1d_for_X_smaller_Y():
    """Test typical values for calculate_required_1d_padding() when X/Y < 1 and has a remainder."""
    actual = [
        calculate_required_1d_padding(X=3, Y=5, frac=-1),
        calculate_required_1d_padding(X=3, Y=5, frac=0),
        calculate_required_1d_padding(X=3, Y=5, frac=1),
        calculate_required_1d_padding(X=3, Y=5, frac=2),
        calculate_required_1d_padding(X=3, Y=5, frac=3),
        calculate_required_1d_padding(X=3, Y=5, frac=4),
    ]

    desired = [2, 2, 2, 2, 1, 1]

    assert actual == desired
