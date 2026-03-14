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

/* Create a chunked file with layout v3 (B-tree v1 index).
 * Uses V18 bounds so the layout message is version 3. */
static void create_chunked_btree_v1(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* V18 bounds → superblock v2, layout message v3, B-tree v1 chunk index */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 1D dataset: 12 int32 values, chunked with chunk size 4 → 3 chunks */
    hsize_t dims[1] = {12};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {4};
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_deflate(dcpl, 4);

    hid_t dset = H5Dcreate2(file, "chunked_v3", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int32_t vals[12];
    for (int i = 0; i < 12; i++) vals[i] = (i + 1) * 10;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked file with one unlimited dimension → extensible array index. */
static void create_extensible_array(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* V110 bounds → layout v4 → extensible array for 1 unlimited dim */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 1D dataset with unlimited max, current size 15, chunk size 5 → 3 chunks */
    hsize_t dims[1] = {15};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {5};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    hid_t dset = H5Dcreate2(file, "extarray", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int32_t vals[15];
    for (int i = 0; i < 15; i++) vals[i] = (i + 1) * 100;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked dataset with implicit index (no filters, early alloc, fixed max). */
static void create_implicit_chunks(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {8};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {4};
    H5Pset_chunk(dcpl, 1, chunk_dims);
    /* Early allocation + no filters → implicit chunk index */
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);

    hid_t dset = H5Dcreate2(file, "implicit", H5T_IEEE_F64LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    double vals[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a 2D chunked dataset where dims don't evenly divide by chunk size. */
static void create_edge_chunks(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 7x5 dataset, chunked 4x3 → 2x2 grid with edge chunks */
    hsize_t dims[2] = {7, 5};
    hid_t space = H5Screate_simple(2, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {4, 3};
    H5Pset_chunk(dcpl, 2, chunk_dims);

    hid_t dset = H5Dcreate2(file, "edge", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* Fill with row-major sequential values: 0, 1, 2, ..., 34 */
    int32_t vals[35];
    for (int i = 0; i < 35; i++) vals[i] = i;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with a compound datatype dataset. */
static void create_compound(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Define compound type: { int32_t id; float x; float y; } = 12 bytes */
    typedef struct {
        int32_t id;
        float x;
        float y;
    } Point;

    hid_t point_type = H5Tcreate(H5T_COMPOUND, sizeof(Point));
    H5Tinsert(point_type, "id", HOFFSET(Point, id), H5T_STD_I32LE);
    H5Tinsert(point_type, "x", HOFFSET(Point, x), H5T_IEEE_F32LE);
    H5Tinsert(point_type, "y", HOFFSET(Point, y), H5T_IEEE_F32LE);

    hsize_t dims[1] = {3};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "points", point_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    Point data[3] = {
        {1, 1.0f, 2.0f},
        {2, 3.0f, 4.0f},
        {3, 5.0f, 6.0f},
    };
    H5Dwrite(dset, point_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Tclose(point_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with an enum datatype dataset. */
static void create_enum(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Define enum type based on int8_t */
    hid_t enum_type = H5Tenum_create(H5T_STD_I8LE);
    int8_t val;
    val = 0; H5Tenum_insert(enum_type, "RED", &val);
    val = 1; H5Tenum_insert(enum_type, "GREEN", &val);
    val = 2; H5Tenum_insert(enum_type, "BLUE", &val);

    hsize_t dims[1] = {5};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "colors", enum_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int8_t data[5] = {0, 1, 2, 1, 0}; /* RED, GREEN, BLUE, GREEN, RED */
    H5Dwrite(dset, enum_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Tclose(enum_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with an array datatype dataset. */
static void create_array(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Array type: int32[3] — each element of the dataset is a 3-element array */
    hsize_t array_dims[1] = {3};
    hid_t array_type = H5Tarray_create2(H5T_STD_I32LE, 1, array_dims);

    /* Dataset: 4 elements, each a 3-element array → 12 total int32 values */
    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "vectors", array_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* 4 vectors: [1,2,3], [4,5,6], [7,8,9], [10,11,12] */
    int32_t data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    H5Dwrite(dset, array_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Tclose(array_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with a Fletcher32 checksum filter. */
static void create_fletcher32(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {10};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {10};
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_fletcher32(dcpl);

    hid_t dset = H5Dcreate2(file, "checksummed", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int32_t vals[10];
    for (int i = 0; i < 10; i++) vals[i] = (i + 1) * 100;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with variable-length string dataset. */
static void create_vlen_strings(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Variable-length string type */
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, H5T_VARIABLE);
    H5Tset_cset(str_type, H5T_CSET_UTF8);

    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "names", str_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    const char *data[4] = {"hello", "world", "HDF5", "variable-length"};
    H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Tclose(str_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file with variable-length integer sequence dataset. */
static void create_vlen_sequence(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Variable-length sequence of int32 */
    hid_t vlen_type = H5Tvlen_create(H5T_STD_I32LE);

    hsize_t dims[1] = {3};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "sequences", vlen_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Three sequences of different lengths */
    int32_t seq0[] = {10, 20};
    int32_t seq1[] = {100, 200, 300, 400};
    int32_t seq2[] = {42};
    hvl_t data[3];
    data[0].len = 2; data[0].p = seq0;
    data[1].len = 4; data[1].p = seq1;
    data[2].len = 1; data[2].p = seq2;

    H5Dwrite(dset, vlen_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Tclose(vlen_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset with an explicit fill value. */
static void create_fill_value(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {6};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    int32_t fill = -999;
    H5Pset_fill_value(dcpl, H5T_NATIVE_INT, &fill);

    hid_t dset = H5Dcreate2(file, "filled", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* Write only first 4 elements; last 2 should get fill value if space allocated */
    int32_t vals[6] = {10, 20, 30, 40, -999, -999};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset with dense attribute storage (many attributes). */
static void create_dense_attributes(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create a group creation property list that triggers dense storage early */
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    /* min_dense=3, max_compact=3: switch to dense after 3 attrs */
    H5Pset_attr_phase_change(gcpl, 3, 3);

    hid_t grp = H5Gcreate2(file, "densegroup", H5P_DEFAULT, gcpl, H5P_DEFAULT);

    hid_t attr_space = H5Screate(H5S_SCALAR);
    char attr_name[64];
    for (int i = 0; i < 8; i++) {
        snprintf(attr_name, sizeof(attr_name), "attr_%02d", i);
        hid_t attr = H5Acreate2(grp, attr_name, H5T_STD_I32LE, attr_space,
                                 H5P_DEFAULT, H5P_DEFAULT);
        int32_t val = (i + 1) * 100;
        H5Awrite(attr, H5T_NATIVE_INT, &val);
        H5Aclose(attr);
    }

    H5Sclose(attr_space);
    H5Gclose(grp);
    H5Pclose(gcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked dataset with multiple unlimited dims → B-tree v2 chunk index. */
static void create_btree_v2_chunks(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* V110 bounds + multiple unlimited dims → B-tree v2 chunk index */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 2D dataset: 6x4, both dims unlimited, chunked 3x2 */
    hsize_t dims[2] = {6, 4};
    hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    hid_t space = H5Screate_simple(2, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {3, 2};
    H5Pset_chunk(dcpl, 2, chunk_dims);

    hid_t dset = H5Dcreate2(file, "bt2chunked", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* Fill with row-major sequential values: 0..23 */
    int32_t vals[24];
    for (int i = 0; i < 24; i++) vals[i] = i;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked dataset with SZIP compression. */
static void create_szip_compressed(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 1D dataset: 32 int32 values, SZIP-compressed */
    hsize_t dims[1] = {32};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {32};
    H5Pset_chunk(dcpl, 1, chunk_dims);
    /* SZIP: NN mode, 8 pixels per block */
    H5Pset_szip(dcpl, H5_SZIP_NN_OPTION_MASK, 8);

    hid_t dset = H5Dcreate2(file, "szip_data", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int32_t vals[32];
    for (int i = 0; i < 32; i++) vals[i] = i * i;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset with a committed (named) datatype. */
static void create_committed_datatype(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create and commit a named datatype */
    hid_t dtype = H5Tcopy(H5T_STD_I32LE);
    H5Tcommit2(file, "mytype", dtype, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create two datasets that share the committed type */
    hsize_t dims[1] = {5};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t ds1 = H5Dcreate2(file, "data1", dtype, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t ds2 = H5Dcreate2(file, "data2", dtype, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int32_t vals1[5] = {10, 20, 30, 40, 50};
    int32_t vals2[5] = {100, 200, 300, 400, 500};
    H5Dwrite(ds1, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals1);
    H5Dwrite(ds2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals2);

    H5Dclose(ds2);
    H5Dclose(ds1);
    H5Sclose(space);
    H5Tclose(dtype);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked dataset large enough that EA data blocks are needed
 * (elements overflow the index block). */
static void create_ea_large(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 1D dataset: 100 elements, chunk size 4 → 25 chunks.
     * With one unlimited dim → extensible array index.
     * The default idx_blk_elmts is typically 2-4, so most chunks
     * will be in data blocks. */
    hsize_t dims[1] = {100};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {4};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    hid_t dset = H5Dcreate2(file, "large_ea", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t vals[100];
    for (int i = 0; i < 100; i++) vals[i] = i * 10;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset with a committed type, plus attributes using that same
 * committed type so we exercise shared-in-attributes resolution. */
static void create_shared_attr(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Commit a named type */
    hid_t dtype = H5Tcopy(H5T_STD_I32LE);
    H5Tcommit2(file, "shared_i32", dtype, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create a dataset using the committed type */
    hsize_t dims[1] = {3};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", dtype, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int32_t vals[3] = {11, 22, 33};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    /* Add an attribute using the same committed type */
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(dset, "scale", dtype, attr_space,
                             H5P_DEFAULT, H5P_DEFAULT);
    int32_t scale = 42;
    H5Awrite(attr, H5T_NATIVE_INT, &scale);
    H5Aclose(attr);
    H5Sclose(attr_space);

    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(dtype);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a chunked dataset that is allocated but never written to —
 * the chunk index address is undefined. */
static void create_empty_chunked(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {10};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {5};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    /* Create but do NOT write — chunk index address should be UNDEF */
    hid_t dset = H5Dcreate2(file, "empty", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a big-endian dataset for testing byte-swap on read. */
static void create_big_endian(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {6};
    hid_t space = H5Screate_simple(1, dims, NULL);

    /* Store as big-endian int32 */
    hid_t dset = H5Dcreate2(file, "be_data", H5T_STD_I32BE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int32_t vals[6] = {1, 256, 65536, -1, 1000000, 0};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
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
    create_chunked_btree_v1("chunked_btree_v1.h5");
    create_extensible_array("extensible_array.h5");
    create_implicit_chunks("implicit_chunks.h5");
    create_edge_chunks("edge_chunks.h5");
    create_compound("compound.h5");
    create_enum("enum.h5");
    create_array("array.h5");
    create_fletcher32("fletcher32.h5");
    create_vlen_strings("vlen_strings.h5");
    create_vlen_sequence("vlen_sequence.h5");
    create_fill_value("fill_value.h5");
    create_dense_attributes("dense_attributes.h5");
    create_btree_v2_chunks("btree_v2_chunks.h5");
    create_szip_compressed("szip_compressed.h5");
    create_big_endian("big_endian.h5");
    create_committed_datatype("committed_datatype.h5");
    create_ea_large("ea_large.h5");
    create_shared_attr("shared_attr.h5");
    create_empty_chunked("empty_chunked.h5");
    return 0;
}
