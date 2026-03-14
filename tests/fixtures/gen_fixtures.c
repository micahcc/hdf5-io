/*
 * Generate HDF5 test fixture files for hdf5-reader.
 * Compile: h5cc -o gen_fixtures gen_fixtures.c
 * Run:     ./gen_fixtures
 */
#include "hdf5.h"
#include <stdio.h>
#include <string.h>

/* Create a minimal file with superblock v2, one contiguous dataset of f64. */
static void create_simple_contiguous(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* Force superblock v2 (libver_bounds = V18..V18) */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create a 4-element 1D dataset of float64 */
    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_IEEE_F64LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    double values[4] = {1.0, 2.0, 3.0, 4.0};
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    /* Add a string attribute */
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(attr_type, 5);
    H5Tset_strpad(attr_type, H5T_STR_NULLTERM);
    hid_t attr = H5Acreate2(dset, "units", attr_type, attr_space,
                             H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, attr_type, "m/s\0\0");

    H5Aclose(attr);
    H5Tclose(attr_type);
    H5Sclose(attr_space);
    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with superblock v3 (SWMR-capable), chunked + deflate. */
static void create_chunked_compressed(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* Force superblock v3 (libver_bounds = V110..V110) */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create a 10x10 chunked, deflate-compressed dataset of int32 */
    hsize_t dims[2] = {10, 10};
    hid_t space = H5Screate_simple(2, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {5, 5};
    H5Pset_chunk(dcpl, 2, chunk_dims);
    H5Pset_deflate(dcpl, 6);

    hid_t dset = H5Dcreate2(file, "compressed", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t values[100];
    for (int i = 0; i < 100; i++) values[i] = i;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with nested groups and multiple datasets. */
static void create_nested_groups(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* /group1 */
    hid_t g1 = H5Gcreate2(file, "group1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* /group1/subgroup */
    hid_t g2 = H5Gcreate2(g1, "subgroup", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* /group1/subgroup/temps - a small dataset */
    hsize_t dims[1] = {3};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(g2, "temps", H5T_IEEE_F32LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float temps[3] = {20.5f, 21.0f, 19.8f};
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temps);
    H5Dclose(dset);
    H5Sclose(space);

    /* /group1/ids - a uint8 dataset */
    hsize_t dims2[1] = {5};
    space = H5Screate_simple(1, dims2, NULL);
    dset = H5Dcreate2(g1, "ids", H5T_STD_U8LE, space,
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    uint8_t ids[5] = {10, 20, 30, 40, 50};
    H5Dwrite(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ids);
    H5Dclose(dset);
    H5Sclose(space);

    H5Gclose(g2);
    H5Gclose(g1);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with a compact dataset (data stored in the object header). */
static void create_compact(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(dcpl, H5D_COMPACT);

    hid_t dset = H5Dcreate2(file, "small", H5T_STD_I16LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int16_t vals[4] = {100, 200, 300, 400};
    H5Dwrite(dset, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with shuffle + deflate filters. */
static void create_shuffle_deflate(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {20};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {20};
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_shuffle(dcpl);
    H5Pset_deflate(dcpl, 4);

    hid_t dset = H5Dcreate2(file, "shuffled", H5T_IEEE_F32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    float vals[20];
    for (int i = 0; i < 20; i++) vals[i] = (float)i * 1.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

int main(void)
{
    create_simple_contiguous("simple_contiguous_v2.h5");
    create_chunked_compressed("chunked_deflate_v3.h5");
    create_nested_groups("nested_groups_v2.h5");
    create_compact("compact_v2.h5");
    create_shuffle_deflate("shuffle_deflate_v3.h5");
    return 0;
}
