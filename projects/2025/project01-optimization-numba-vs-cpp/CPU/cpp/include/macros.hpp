#ifndef MACROS_HPP
#define MACROS_HPP

#define AT(arr, i, j, stride) (arr[(i) * (stride) + (j)])

#define ALIGN_BYTES 64 // Default alignment for SIMD

// Tile sizes for blocking
#define TILE_I 32
#define TILE_J 32

#define T_INNER 4 // Temporal block size for inner loops

#endif // MACROS_HPP