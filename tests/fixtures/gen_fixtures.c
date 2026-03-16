/*
 * Generate HDF5 test fixture files for hdf5-reader.
 * Compile: h5cc -o gen_fixtures gen_fixtures.c
 * Run:     ./gen_fixtures
 */
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Minimal LZF compression/decompression (Marc Lehmann's algorithm) ──
 * Used to register as HDF5 third-party filter ID 32000 (same as h5py). */

#define LZF_HLOG 16
#define LZF_HTAB_SIZE (1 << LZF_HLOG)

static unsigned int
lzf_compress(const void *const in_data, unsigned int in_len,
             void *out_data, unsigned int out_len)
{
    const unsigned char *ip = (const unsigned char *)in_data;
    unsigned char *op = (unsigned char *)out_data;
    const unsigned char *in_end = ip + in_len;
    unsigned char *out_end = op + out_len;
    const unsigned char *ref;

    unsigned int *htab;
    htab = (unsigned int *)calloc(LZF_HTAB_SIZE, sizeof(unsigned int));
    if (!htab) return 0;

    const unsigned char *lit_start = ip;
    unsigned char *lit_op = op++; /* placeholder for first literal ctrl */

    if (in_len < 4) goto literal_flush;
    ip++;

    for (;;) {
        if (ip >= in_end - 2) break;

        unsigned int v = ip[0] | ((unsigned int)ip[1] << 8) | ((unsigned int)ip[2] << 16);
        unsigned int h = ((v >> 1) ^ v);
        h = (h >> 3 ^ h << 5 ^ h >> 12) & (LZF_HTAB_SIZE - 1);
        ref = (const unsigned char *)in_data + htab[h];
        htab[h] = (unsigned int)(ip - (const unsigned char *)in_data);

        unsigned int off = (unsigned int)(ip - ref);
        if (off > 0 && off < 8192 && ref >= (const unsigned char *)in_data
            && ref + 2 < in_end
            && ref[0] == ip[0] && ref[1] == ip[1] && ref[2] == ip[2])
        {
            /* Flush pending literals */
            unsigned int lit_len = (unsigned int)(ip - lit_start);
            if (lit_len > 0) {
                const unsigned char *lp = lit_start;
                while (lp < ip) {
                    unsigned int run = (unsigned int)(ip - lp);
                    if (run > 32) run = 32;
                    if (op + 1 + run > out_end) { free(htab); return 0; }
                    *lit_op = (unsigned char)(run - 1);
                    memcpy(op, lp, run);
                    op += run;
                    lp += run;
                    if (lp < ip) lit_op = op++;
                }
            }

            /* Find match length */
            unsigned int len = 3;
            unsigned int maxlen = (unsigned int)(in_end - ip);
            if (maxlen > 264) maxlen = 264;
            while (len < maxlen && ref[len] == ip[len]) len++;

            unsigned int offset = off - 1;
            if (len <= 8) {
                if (op + 2 > out_end) { free(htab); return 0; }
                *op++ = (unsigned char)(((len - 2) << 5) | (offset >> 8));
                *op++ = (unsigned char)(offset & 0xFF);
            } else {
                if (op + 3 > out_end) { free(htab); return 0; }
                *op++ = (unsigned char)((7 << 5) | (offset >> 8));
                *op++ = (unsigned char)(len - 9);
                *op++ = (unsigned char)(offset & 0xFF);
            }

            ip += len;
            lit_start = ip;
            if (op >= out_end) { free(htab); return 0; }
            lit_op = op++;

            if (ip < in_end - 2) {
                v = ip[0] | ((unsigned int)ip[1] << 8) | ((unsigned int)ip[2] << 16);
                h = ((v >> 1) ^ v);
                h = (h >> 3 ^ h << 5 ^ h >> 12) & (LZF_HTAB_SIZE - 1);
                htab[h] = (unsigned int)(ip - (const unsigned char *)in_data);
            }
        } else {
            ip++;
        }
    }

literal_flush:;
    /* Flush remaining */
    if (lit_start < in_end) {
        const unsigned char *lp = lit_start;
        const unsigned char *end = in_end;
        while (lp < end) {
            unsigned int run = (unsigned int)(end - lp);
            if (run > 32) run = 32;
            if (op + 1 + run > out_end) { free(htab); return 0; }
            *lit_op = (unsigned char)(run - 1);
            memcpy(op, lp, run);
            op += run;
            lp += run;
            if (lp < end) lit_op = op++;
        }
    } else {
        /* No remaining literals — remove unused ctrl placeholder */
        op--;
    }

    free(htab);
    return (unsigned int)(op - (unsigned char *)out_data);
}

static unsigned int
lzf_decompress(const void *const in_data, unsigned int in_len,
               void *out_data, unsigned int out_len)
{
    const unsigned char *ip = (const unsigned char *)in_data;
    unsigned char *op = (unsigned char *)out_data;
    const unsigned char *in_end = ip + in_len;
    unsigned char *out_end = op + out_len;

    while (ip < in_end) {
        unsigned int ctrl = *ip++;
        if (ctrl < 32) {
            unsigned int len = ctrl + 1;
            if (ip + len > in_end || op + len > out_end) return 0;
            memcpy(op, ip, len);
            ip += len;
            op += len;
        } else {
            unsigned int len = (ctrl >> 5) + 2;
            if (len == 9) {
                if (ip >= in_end) return 0;
                len += *ip++;
            }
            if (ip >= in_end) return 0;
            unsigned int offset = ((ctrl & 0x1f) << 8) | *ip++;
            unsigned char *ref = op - offset - 1;
            if (ref < (unsigned char *)out_data || op + len > out_end) return 0;
            /* byte-by-byte copy (may overlap) */
            for (unsigned int i = 0; i < len; i++) *op++ = ref[i];
        }
    }
    return (unsigned int)(op - (unsigned char *)out_data);
}

/* HDF5 filter callback for LZF (filter ID 32000). */
#define H5Z_FILTER_LZF 32000

static size_t
lzf_filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
           size_t nbytes, size_t *buf_size, void **buf)
{
    if (flags & H5Z_FLAG_REVERSE) {
        /* Decompress */
        size_t out_size = cd_nelmts >= 3 ? cd_values[2] : nbytes * 4;
        void *outbuf = malloc(out_size);
        if (!outbuf) return 0;
        unsigned int result = lzf_decompress(*buf, (unsigned int)nbytes,
                                              outbuf, (unsigned int)out_size);
        if (result == 0) { free(outbuf); return 0; }
        free(*buf);
        *buf = outbuf;
        *buf_size = out_size;
        return (size_t)result;
    } else {
        /* Compress */
        size_t out_max = nbytes + (nbytes / 16) + 64 + 3;
        void *outbuf = malloc(out_max);
        if (!outbuf) return 0;
        unsigned int result = lzf_compress(*buf, (unsigned int)nbytes,
                                            outbuf, (unsigned int)out_max);
        if (result == 0 || result >= nbytes) {
            /* Compression didn't help — store uncompressed */
            free(outbuf);
            return nbytes;
        }
        free(*buf);
        *buf = outbuf;
        *buf_size = out_max;
        return (size_t)result;
    }
}

static void register_lzf_filter(void)
{
    H5Z_class2_t filter_class = {
        .version = H5Z_CLASS_T_VERS,
        .id = H5Z_FILTER_LZF,
        .encoder_present = 1,
        .decoder_present = 1,
        .name = "lzf",
        .can_apply = NULL,
        .set_local = NULL,
        .filter = lzf_filter,
    };
    H5Zregister(&filter_class);
}

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

/* Create a group with creation order tracked & indexed for both links and attributes.
 * Links and attributes are created in non-alphabetical order so creation order
 * differs from name order. Dense storage is forced by setting phase change to 0. */
static void create_creation_order(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Group creation property: track+index creation order, force dense storage */
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    H5Pset_link_phase_change(gcpl, 0, 0);
    H5Pset_attr_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    H5Pset_attr_phase_change(gcpl, 0, 0);

    hid_t grp = H5Gcreate2(file, "ordered", H5P_DEFAULT, gcpl, H5P_DEFAULT);

    /* Create child groups in non-alphabetical order: charlie, alpha, bravo */
    hid_t g1 = H5Gcreate2(grp, "charlie", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(g1);
    hid_t g2 = H5Gcreate2(grp, "alpha", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(g2);
    hid_t g3 = H5Gcreate2(grp, "bravo", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(g3);

    /* Create attributes in non-alphabetical order: zebra, mango, apple */
    hid_t attr_space = H5Screate(H5S_SCALAR);
    int32_t val;

    val = 30;
    hid_t a1 = H5Acreate2(grp, "zebra", H5T_STD_I32LE, attr_space,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a1, H5T_NATIVE_INT, &val);
    H5Aclose(a1);

    val = 10;
    hid_t a2 = H5Acreate2(grp, "mango", H5T_STD_I32LE, attr_space,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a2, H5T_NATIVE_INT, &val);
    H5Aclose(a2);

    val = 20;
    hid_t a3 = H5Acreate2(grp, "apple", H5T_STD_I32LE, attr_space,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a3, H5T_NATIVE_INT, &val);
    H5Aclose(a3);

    H5Sclose(attr_space);
    H5Gclose(grp);
    H5Pclose(gcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset of complex double values (HDF5 2.0 native complex type). */
static void create_complex(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* Complex types require HDF5 2.0 file format */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V200, H5F_LIBVER_V200);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* 4-element 1D dataset of complex double */
    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "complex_data", H5T_COMPLEX_IEEE_F64LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Values: (1+2i), (3+4i), (-1+0i), (0-5i)
     * On disk, complex is just two consecutive doubles (real, imag).
     * Use the file type for both mem and file to write raw doubles. */
    double values[8] = {1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.0, -5.0};
    H5Dwrite(dset, H5T_COMPLEX_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* N-bit filter: uint16 with 10-bit precision */
static void create_nbit(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create datatype: uint16 with 10-bit precision */
    hid_t dtype = H5Tcopy(H5T_NATIVE_UINT16);
    H5Tset_precision(dtype, 10);
    H5Tset_offset(dtype, 0);

    hsize_t dims[1] = {8};
    hsize_t chunk_dims[1] = {8};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_nbit(dcpl);

    hid_t dset = H5Dcreate2(file, "data", dtype, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    uint16_t values[8] = {0, 100, 200, 300, 400, 500, 600, 700};
    H5Dwrite(dset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(dtype);
    H5Pclose(dcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Scale-offset filter: int32 with SO_INT */
static void create_scaleoffset(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {8};
    hsize_t chunk_dims[1] = {8};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    H5Pset_scaleoffset(dcpl, H5Z_SO_INT, H5Z_SO_INT_MINBITS_DEFAULT);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT32, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t values[8] = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007};
    H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Filtered fractal heap: group with deflate on dense link/attribute storage */
static void create_filtered_fheap(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create gcpl with deflate and force dense storage immediately (thresholds 0,0) */
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    H5Pset_link_phase_change(gcpl, 0, 0);
    H5Pset_deflate(gcpl, 6);

    /* Create group with filtered fractal heap */
    hid_t grp = H5Gcreate2(file, "filtered_group", H5P_DEFAULT, gcpl, H5P_DEFAULT);

    /* Add many soft links to potentially force indirect blocks */
    char lname[32];
    for (int i = 0; i < 30; i++) {
        snprintf(lname, sizeof(lname), "link_%03d", i);
        H5Lcreate_soft("/", grp, lname, H5P_DEFAULT, H5P_DEFAULT);
    }

    /* Also add a hard-linked dataset so we can verify data access through the group */
    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(grp, "ds", H5T_IEEE_F64LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double values[4] = {10.0, 20.0, 30.0, 40.0};
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    /* Add attributes to the group (also in the filtered fractal heap) */
    hid_t ascalar = H5Screate(H5S_SCALAR);
    hid_t a1 = H5Acreate2(grp, "attr_one", H5T_NATIVE_INT32, ascalar,
                           H5P_DEFAULT, H5P_DEFAULT);
    int32_t v1 = 42;
    H5Awrite(a1, H5T_NATIVE_INT32, &v1);
    H5Aclose(a1);

    hid_t a2 = H5Acreate2(grp, "attr_two", H5T_NATIVE_INT32, ascalar,
                           H5P_DEFAULT, H5P_DEFAULT);
    int32_t v2 = 99;
    H5Awrite(a2, H5T_NATIVE_INT32, &v2);
    H5Aclose(a2);
    H5Sclose(ascalar);

    H5Dclose(dset);
    H5Sclose(space);
    H5Gclose(grp);
    H5Pclose(gcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a B-tree v2 chunked dataset with enough chunks to force depth > 0.
 * 2D dataset, both dims unlimited → B-tree v2.
 * Shape 20x10, chunk 1x1 → 200 chunks.
 * record_size = 8(addr) + 2*8(scaled) = 24
 * max_leaf_records = (4096 - 10) / 24 = 170, so 200 chunks → depth=1 */
static void create_btree_v2_deep(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[2] = {20, 10};
    hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    hid_t space = H5Screate_simple(2, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {1, 1};
    H5Pset_chunk(dcpl, 2, chunk_dims);

    hid_t dset = H5Dcreate2(file, "deep", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t vals[200];
    for (int i = 0; i < 200; i++) vals[i] = i;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a B-tree v2 chunked dataset with deflate filter.
 * 2D dataset, both dims unlimited → B-tree v2.
 * Shape 6x4, chunk 3x2 → 4 chunks. Exercises filtered record field order
 * and chunk_size_len formula. */
static void create_btree_v2_filtered(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[2] = {6, 4};
    hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    hid_t space = H5Screate_simple(2, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[2] = {3, 2};
    H5Pset_chunk(dcpl, 2, chunk_dims);
    H5Pset_deflate(dcpl, 6);

    hid_t dset = H5Dcreate2(file, "filtered", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

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

/* Create a group with enough child groups to force the fractal heap root
 * to become an indirect block (exercises indirect block traversal). */
static void create_fheap_indirect(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Force dense link storage immediately */
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    H5Pset_link_phase_change(gcpl, 0, 0);

    hid_t grp = H5Gcreate2(file, "many", H5P_DEFAULT, gcpl, H5P_DEFAULT);

    /* Create 120 child groups with long names to overflow initial direct block */
    char name[64];
    for (int i = 0; i < 120; i++) {
        snprintf(name, sizeof(name), "child_group_%04d_padding", i);
        hid_t child = H5Gcreate2(grp, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(child);
    }

    /* Also add a dataset so we can verify data access through the group */
    hsize_t dims[1] = {5};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(grp, "values", H5T_STD_I32LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int32_t vals[5] = {10, 20, 30, 40, 50};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    H5Dclose(dset);
    H5Sclose(space);
    H5Gclose(grp);
    H5Pclose(gcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a dataset with enough attributes that the object header contains
 * nil (type 0) padding messages between real messages, exercising the nil
 * message skip path in the object header parser. */
static void create_nil_messages(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Dataset with small data but many attributes to fill the header chunk */
    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_STD_I32LE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int32_t vals[4] = {1, 2, 3, 4};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    /* Add attributes with varying name lengths to create uneven alignment
     * and force nil message padding in the object header */
    hid_t scalar = H5Screate(H5S_SCALAR);
    char aname[64];
    for (int i = 0; i < 12; i++) {
        snprintf(aname, sizeof(aname), "attribute_%d", i);
        hid_t attr = H5Acreate2(dset, aname, H5T_STD_I32LE, scalar,
                                 H5P_DEFAULT, H5P_DEFAULT);
        int32_t v = (i + 1) * 10;
        H5Awrite(attr, H5T_NATIVE_INT, &v);
        H5Aclose(attr);
    }

    H5Sclose(scalar);
    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a vlen string dataset, then delete and re-create entries to
 * produce global heap collections with free-space (index 0) entries
 * interspersed with valid objects. */
static void create_gcol_free_space(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* First create a vlen string dataset and write it */
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, H5T_VARIABLE);

    hsize_t dims[1] = {4};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "strings", str_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write initial values (long strings to use space) */
    const char *data1[4] = {
        "this_is_a_long_string_alpha",
        "this_is_a_long_string_bravo",
        "short",
        "this_is_a_long_string_delta"
    };
    H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data1);

    /* Overwrite with shorter strings to potentially create free space in
     * the global heap collection */
    const char *data2[4] = {"hi", "there", "hdf5", "test"};
    H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2);

    H5Dclose(dset);
    H5Tclose(str_type);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a compound type containing enum, array, and plain int32 members.
 * Exercises encoded_size computation for Enum and Array within compound types.
 * Without the encoded_size fix, the parser would consume all remaining bytes
 * when computing the Enum/Array member's size, failing to parse subsequent members. */
static void create_compound_complex_members(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Create an enum type (base int32, 3 values) */
    hid_t enum_type = H5Tenum_create(H5T_STD_I32LE);
    int32_t eval;
    eval = 0; H5Tenum_insert(enum_type, "RED", &eval);
    eval = 1; H5Tenum_insert(enum_type, "GREEN", &eval);
    eval = 2; H5Tenum_insert(enum_type, "BLUE", &eval);

    /* Create an array type: int32[3] */
    hsize_t arr_dims[1] = {3};
    hid_t arr_type = H5Tarray_create2(H5T_STD_I32LE, 1, arr_dims);

    /* Build compound: { int32 color (enum); int32 coords[3] (array); int32 id; }
     * = 4 + 12 + 4 = 20 bytes */
    hid_t comp = H5Tcreate(H5T_COMPOUND, 20);
    H5Tinsert(comp, "color", 0, enum_type);
    H5Tinsert(comp, "coords", 4, arr_type);
    H5Tinsert(comp, "id", 16, H5T_STD_I32LE);

    hsize_t dims[1] = {3};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "records", comp, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write 3 records */
    uint8_t data[60]; /* 3 * 20 = 60 */
    memset(data, 0, sizeof(data));
    /* Record 0: color=RED(0), coords=[10,20,30], id=100 */
    int32_t tmp;
    tmp = 0;   memcpy(data + 0, &tmp, 4);
    tmp = 10;  memcpy(data + 4, &tmp, 4);
    tmp = 20;  memcpy(data + 8, &tmp, 4);
    tmp = 30;  memcpy(data + 12, &tmp, 4);
    tmp = 100; memcpy(data + 16, &tmp, 4);
    /* Record 1: color=GREEN(1), coords=[40,50,60], id=200 */
    tmp = 1;   memcpy(data + 20, &tmp, 4);
    tmp = 40;  memcpy(data + 24, &tmp, 4);
    tmp = 50;  memcpy(data + 28, &tmp, 4);
    tmp = 60;  memcpy(data + 32, &tmp, 4);
    tmp = 200; memcpy(data + 36, &tmp, 4);
    /* Record 2: color=BLUE(2), coords=[70,80,90], id=300 */
    tmp = 2;   memcpy(data + 40, &tmp, 4);
    tmp = 70;  memcpy(data + 44, &tmp, 4);
    tmp = 80;  memcpy(data + 48, &tmp, 4);
    tmp = 90;  memcpy(data + 52, &tmp, 4);
    tmp = 300; memcpy(data + 56, &tmp, 4);

    H5Dwrite(dset, comp, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(comp);
    H5Tclose(arr_type);
    H5Tclose(enum_type);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Create a file exercising SWMR (Single Writer Multiple Reader) mode.
 * SWMR requires superblock v3 and chunked datasets.
 * The file is created normally, then reopened in SWMR write mode where
 * the dataset is extended and new data is written. After clean close the
 * SWMR flags are cleared, but the file retains superblock v3 layout and
 * data written during the SWMR session. */
/* LZF-compressed chunked dataset (third-party filter ID 32000).
 * Simple: 32 i32 values in a single chunk, V18 superblock, layout v3 (B-tree v1). */
static void create_lzf(const char *filename)
{
    register_lzf_filter();

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {32};
    hsize_t chunk_dims[1] = {32};
    hid_t space = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);
    /* Add LZF filter (ID 32000) with no client data */
    H5Pset_filter(dcpl, H5Z_FILTER_LZF, H5Z_FLAG_OPTIONAL, 0, NULL);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT32, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t values[32];
    for (int i = 0; i < 32; i++) values[i] = i / 4;  /* repeated values → compressible */
    H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(file);
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

static void create_swmr(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    /* Phase 1: Create file with chunked dataset + attribute */
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t dims[1] = {4};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {4};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    hid_t dset = H5Dcreate2(file, "data", H5T_STD_I32LE, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int32_t vals[4] = {10, 20, 30, 40};
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals);

    /* Attribute must be created before SWMR mode */
    hid_t ascalar = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(dset, "scale", H5T_NATIVE_INT32, ascalar,
                             H5P_DEFAULT, H5P_DEFAULT);
    int32_t scale = 100;
    H5Awrite(attr, H5T_NATIVE_INT32, &scale);
    H5Aclose(attr);
    H5Sclose(ascalar);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);

    /* Phase 2: Reopen in SWMR write mode and extend dataset */
    file = H5Fopen(filename, H5F_ACC_RDWR | H5F_ACC_SWMR_WRITE, fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    hsize_t new_dims[1] = {8};
    H5Dset_extent(dset, new_dims);

    /* Write to the extended portion [4..8) */
    hid_t fspace = H5Dget_space(dset);
    hsize_t start[1] = {4};
    hsize_t count[1] = {4};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t mspace = H5Screate_simple(1, count, NULL);
    int32_t new_vals[4] = {50, 60, 70, 80};
    H5Dwrite(dset, H5T_NATIVE_INT, mspace, fspace, H5P_DEFAULT, new_vals);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);
    H5Fclose(file);  /* Clean close → SWMR flags cleared */
    H5Pclose(fapl);
    printf("Created %s\n", filename);
}

/* Simpler version of fheap_indirect: 27 groups with longer names
 * so the fractal heap overflows into an indirect block (5+ direct blocks)
 * but the B-tree stays at depth 0 (single leaf, no splits).
 * This avoids the complex interleaved allocation pattern. */
static void create_fheap_indirect_simple(const char *filename)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110);

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* Force dense link storage immediately */
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    H5Pset_link_phase_change(gcpl, 0, 0);

    hid_t grp = H5Gcreate2(file, "many", H5P_DEFAULT, gcpl, H5P_DEFAULT);

    /* 27 child groups with ~64-char names to overflow into 5+ direct blocks
     * while keeping B-tree at depth 0 (27 < 45 max records per leaf) */
    char name[80];
    for (int i = 0; i < 27; i++) {
        snprintf(name, sizeof(name),
                 "child_group_%04d_with_extra_padding_to_make_the_name_longer_end", i);
        hid_t child = H5Gcreate2(grp, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(child);
    }

    H5Gclose(grp);
    H5Pclose(gcpl);
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
    create_creation_order("creation_order.h5");
    create_complex("complex.h5");
    create_nbit("nbit.h5");
    create_scaleoffset("scaleoffset.h5");
    create_filtered_fheap("filtered_fheap.h5");
    create_btree_v2_deep("btree_v2_deep.h5");
    create_btree_v2_filtered("btree_v2_filtered.h5");
    create_fheap_indirect("fheap_indirect.h5");
    create_nil_messages("nil_messages.h5");
    create_gcol_free_space("gcol_free_space.h5");
    create_compound_complex_members("compound_complex_members.h5");
    create_swmr("swmr.h5");
    create_lzf("lzf_c.h5");
    create_fheap_indirect_simple("fheap_indirect_simple.h5");
    return 0;
}
