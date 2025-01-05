//! Rust bindings to [`libdeflate`], a DEFLATE-based buffer
//! compression/decompression library that works with raw DEFLATE,
//! zlib, and gzip data.
//!
//! **Warning**: Libdeflate is targeted at *specialized*
//! performance-sensitive use-cases where developers have a good
//! understanding of their input/output data. Developers looking for a
//! general-purpose DEFLATE library should use something like
//! [`flate2`], which can handle a much wider range of inputs (network
//! streams, large files, etc.).
//!
//! [`libdeflate`]: https://github.com/ebiggers/libdeflate
//! [`flate2`]: https://github.com/alexcrichton/flate2-rs
//!
//! # Decompression
//!
//! [`Decompressor::new`] can be used to construct a [`Decompressor`],
//! which can decompress:
//!
//! - DEFLATE data ([`deflate_decompress`])
//! - zlib data ([`zlib_decompress`])
//! - gzip data ([`gzip_decompress`])
//!
//! **Note**: `libdeflate` requires that the input *and* output
//! buffers are pre-allocated before decompressing. Because of this,
//! you will at least need to know the upper bound on how large the
//! compressed data will decompress to; otherwise, a `decompress_*`
//! function call will return `DecompressionError::InsufficientSpace`
//!
//! [`Decompressor::new`]: struct.Decompressor.html#method.new
//! [`Decompressor`]: struct.Decompressor.html
//! [`deflate_decompress`]: struct.Decompressor.html#method.deflate_decompress
//! [`zlib_decompress`]: struct.Decompressor.html#method.zlib_decompress
//! [`gzip_decompress`]: struct.Decompressor.html#method.gzip_decompress
//! [`DecompressionError::InsufficientSpace`]: enum.DecompressionError.html
//!
//! # Compression
//!
//! `Compressor::new` can be used to construct a [`Compressor`], which
//! can compress data into the following formats:
//!
//! - DEFLATE ([`deflate_compress`])
//! - zlib ([`zlib_compress`])
//! - gzip ([`gzip_compress`])
//!
//! Because buffers must be allocated up-front, developers need to
//! supply these functions with output buffers that are big enough to
//! fit the compressed data. The maximum size of the compressed data
//! can be found with the associated `*_bound` methods:
//!
//! - [`deflate_compress_bound`]
//! - [`zlib_compress_bound`]
//! - [`gzip_compress_bound`]
//!
//! [`Compressor::new`]: struct.Compressor.html#method.new
//! [`Compressor`]: struct.Compressor.html
//! [`deflate_compress`]: struct.Compressor.html#method.deflate_compress
//! [`zlib_compress`]: struct.Compressor.html#method.zlib_compress
//! [`gzip_compress`]: struct.Compressor.html#method.gzip_compress
//! [`deflate_compress_bound`]: struct.Compressor.html#method.deflate_compress_bound
//! [`zlib_compress_bound`]: struct.Compressor.html#method.zlib_compress_bound
//! [`gzip_compress_bound`]: struct.Compressor.html#method.gzip_compress_bound

use byteorder::{LittleEndian, ReadBytesExt};
use libdeflate_sys::{
    libdeflate_adler32, libdeflate_alloc_compressor, libdeflate_alloc_decompressor,
    libdeflate_alloc_gdeflate_compressor, libdeflate_alloc_gdeflate_decompressor,
    libdeflate_compressor, libdeflate_crc32, libdeflate_decompressor, libdeflate_deflate_compress,
    libdeflate_deflate_compress_bound, libdeflate_deflate_decompress, libdeflate_free_compressor,
    libdeflate_free_decompressor, libdeflate_free_gdeflate_compressor,
    libdeflate_free_gdeflate_decompressor, libdeflate_gdeflate_compress,
    libdeflate_gdeflate_compress_bound, libdeflate_gdeflate_compressor,
    libdeflate_gdeflate_decompress, libdeflate_gdeflate_decompressor, libdeflate_gdeflate_in_page,
    libdeflate_gdeflate_out_page, libdeflate_gzip_compress, libdeflate_gzip_compress_bound,
    libdeflate_gzip_decompress, libdeflate_result, libdeflate_result_LIBDEFLATE_BAD_DATA,
    libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE, libdeflate_result_LIBDEFLATE_SUCCESS,
    libdeflate_zlib_compress, libdeflate_zlib_compress_bound, libdeflate_zlib_decompress,
};
use core::num;
use std::fmt::{self, Write};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::ptr::null;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, Thread};
use std::{convert::TryInto, error::Error};

#[cfg(feature = "use_rust_alloc")]
mod malloc_wrapper;
#[cfg(feature = "use_rust_alloc")]
use malloc_wrapper::init_allocator;

#[cfg(not(feature = "use_rust_alloc"))]
fn init_allocator() {}

const K_GDEFLATE_ID: u8 = 4;
const KDEFAULT_TILE_SIZE: usize = 64 * 1024;
const MINIMUM_COMPRESSION_LEVEL: u32 = 1;
const MAXIMUM_COMPRESSION_LEVEL: u32 = 32;

enum Flags {
    CompressSingleThread = 0x200,
}
pub trait BitField<Bits> {
    type Output;
    fn bit_split(self, bits: Bits) -> Self::Output;
}

macro_rules! smash {
    ($a:ident) => {
        usize
    };
}

macro_rules! smash2 {
    ($a:ident, $b:ty) => {
        $b
    };
}

macro_rules! impl_bit {
    ($base_type:ty, $($a:ident),+) => {
        impl BitField<($(smash!($a)),*)> for $base_type {
            type Output = ($(smash2!($a, $base_type)),*);
            #[allow(unused_assignments)]
            fn bit_split(mut self, ($($a),*): ($(smash!($a)),*)) -> Self::Output {
                if 0 $(+$a)* != std::mem::size_of::<$base_type>() * 8 {
                    panic!();
                }
                $(
                    let t = self & ((1 << $a) - 1);
                    self >>= $a;
                    let $a = t;
                )*

                ($($a),*)
            }
        }
    };
}

impl_bit!(u8, a, b);
impl_bit!(u8, a, b, c);
impl_bit!(u8, a, b, c, d);
impl_bit!(u8, a, b, c, d, e);

impl_bit!(u16, a, b);
impl_bit!(u16, a, b, c);
impl_bit!(u16, a, b, c, d);
impl_bit!(u16, a, b, c, d, e);

impl_bit!(u32, a, b);
impl_bit!(u32, a, b, c);
impl_bit!(u32, a, b, c, d);
impl_bit!(u32, a, b, c, d, e);

impl_bit!(u64, a, b);
impl_bit!(u64, a, b, c);
impl_bit!(u64, a, b, c, d);
impl_bit!(u64, a, b, c, d, e);

#[derive(Debug)]
pub struct TileStream {
    id: u8,
    magic: u8,
    num_tiles: u16,      //u16,
    tile_size_idx: u32,  //u8,
    last_tile_size: u32, //u32,
    reserv1: u32,
    //bitfield: u32, //tileSizeIdx: 2, lastTileSize: 18, reserv1: 12
}

impl TileStream {
    pub fn new(uncompressed_size: usize) -> TileStream {
        let mut num_tiles = (uncompressed_size / KDEFAULT_TILE_SIZE).try_into().unwrap();
        let last_tile_size = (uncompressed_size - num_tiles as usize * KDEFAULT_TILE_SIZE)
            .try_into()
            .unwrap();
        num_tiles += if last_tile_size != 0 { 1 } else { 0 };
        TileStream {
            id: K_GDEFLATE_ID,
            magic: K_GDEFLATE_ID ^ 0xff,
            tile_size_idx: 1,
            num_tiles,
            last_tile_size,
            reserv1: 0,
        }
    }

    pub fn from<W: Read + std::io::Seek>(data: &mut W) -> std::io::Result<TileStream> {
        let id = data.read_u8()?;
        let magic = data.read_u8()?;
        let num_tiles = data.read_u16::<LittleEndian>()?;
        let flags = data.read_u32::<LittleEndian>()?;
        let (tile_size_idx, last_tile_size, reserv1) = flags.bit_split((2, 18, 12));
        Ok(TileStream {
            id,
            magic,
            num_tiles,
            tile_size_idx,
            last_tile_size,
            reserv1,
        })
    }

    pub fn get_uncompressed_size(&self) -> usize {
        self.num_tiles as usize * KDEFAULT_TILE_SIZE
            - if self.last_tile_size == 0 {
                0
            } else {
                KDEFAULT_TILE_SIZE - self.last_tile_size as usize
            }
    }

    pub fn is_valid(&self) -> bool {
        self.id == self.magic ^ 0xff && self.id == K_GDEFLATE_ID
    }
}

pub struct GDeflateDecompressor {
    p: *mut libdeflate_gdeflate_decompressor,
}
unsafe impl Send for GDeflateDecompressor {}

impl GDeflateDecompressor {
    pub fn new() -> GDeflateDecompressor {
        unsafe {
            init_allocator();
            let ptr = libdeflate_alloc_gdeflate_decompressor();
            if !ptr.is_null() {
                GDeflateDecompressor { p: ptr }
            } else {
                panic!("libdeflate_alloc_gdeflate_decompressor returned NULL: out of memory")
            }
        }
    }
}

const MAX_WORKERS: u32 = 31;

impl GDeflateDecompressor {
    pub fn gdeflate_decompress_threaded(
        in_data: &[u8],
        out_data: &mut [u8],
        num_workers: u32,
    ) -> DecompressionResult<usize> {
        let bytes_read_res = Arc::new(AtomicUsize::new(0));
        unsafe {
            init_allocator();

            // Read header 
            let mut in_data_cursor = Cursor::new(&in_data);
            let header = TileStream::from(&mut in_data_cursor).unwrap();
            if !header.is_valid() {
                return Err(DecompressionError::BadData);
            }

            let tile_index = Arc::new(AtomicU32::new(0));
            let failed = Arc::new(AtomicBool::new(false));
            let result = Arc::new(Mutex::new(Ok(())));

            // Set up worker threads
            let num_workers = MAX_WORKERS.min(num_workers).max(1);
            let mut worker_handles: Vec<JoinHandle<()>> = Vec::with_capacity(num_workers as usize);

            let in_ptr = in_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as *mut u8 as usize;
            for i in 0..num_workers {
                let tidx = tile_index.clone();
                let num_tiles = header.num_tiles as u32;
                let failed = failed.clone();
                let result = result.clone();
                let bytes_read = bytes_read_res.clone();

                worker_handles.push(std::thread::spawn(move || {
                    let mut worker_bytes_read = 0;
                    let decompressor = GDeflateDecompressor::new();
                    let tile_offsets = (in_ptr as *const u8).add(size_of::<u64>()) as *const u32;
                    let in_data_ptr = tile_offsets.add(num_tiles as usize) as *const u8;
                    loop {
                        // get tile info
                        let tile_index = tidx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if tile_index >= num_tiles || failed.load(Ordering::Relaxed) {
                            break;
                        }
                        let tile_offset = if tile_index > 0 {
                            *tile_offsets.wrapping_add(tile_index as usize)
                        } else {
                            0
                        };

                        // Create the page
                        let data_size = if tile_index < num_tiles - 1 {
                            *tile_offsets.wrapping_add(tile_index as usize + 1) - tile_offset as u32
                        } else {
                            *tile_offsets
                        };
                        let data_ptr = in_data_ptr.add(tile_offset as usize) as *const std::ffi::c_void;
                        let mut compressed_page = libdeflate_gdeflate_in_page {
                            data: data_ptr,
                            nbytes: data_size as usize,
                        };

                        let output_offset = tile_index as usize * KDEFAULT_TILE_SIZE;
                        let out_ptr = (out_ptr as *mut u8).wrapping_add(output_offset) as *mut std::ffi::c_void;

                        let mut out_nbytes = 0;
                        let decomp_result: libdeflate_result = libdeflate_gdeflate_decompress(
                            decompressor.p,
                            &mut compressed_page,
                            1,
                            out_ptr,
                            KDEFAULT_TILE_SIZE,
                            &mut out_nbytes,
                        ) as libdeflate_result;

                        let mut result = result.lock().unwrap();
                        match decomp_result {
                            libdeflate_result_LIBDEFLATE_SUCCESS => worker_bytes_read += out_nbytes,
                            libdeflate_result_LIBDEFLATE_BAD_DATA => {
                                *result = Err(DecompressionError::BadData);
                                failed.store(true, Ordering::Relaxed);
                                break;
                            },
                            libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                                *result = Err(DecompressionError::InsufficientSpace);
                                failed.store(true, Ordering::Relaxed);
                                break;
                            }
                            _ => {
                                panic!("libdeflate_gdeflate_decompress returned an unknown error type: this is an internal bug that **must** be fixed");
                            }
                        }
                    }
                    let x = bytes_read.load(Ordering::Relaxed);
                    bytes_read.store(x + worker_bytes_read, Ordering::Relaxed);
                }));
            }

            for worker in worker_handles {
                worker.join().unwrap();
            }
            let res = result.lock().unwrap();
            match *res {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
        let x = bytes_read_res.load(Ordering::Relaxed);
        Ok(x)
    }
    pub fn gdeflate_decompress(
        in_data_raw: &[u8],
        out_data: &mut [u8],
    ) -> DecompressionResult<usize> {
        let mut bytes_read = 0;
        unsafe {
            init_allocator();
            let mut in_data = Cursor::new(&in_data_raw);
            let header = TileStream::from(&mut in_data).unwrap();
            if !header.is_valid() {
                return Err(DecompressionError::BadData);
            }
            let tile_offsets = in_data_raw.as_ptr().add(size_of::<u64>()) as *const u32;
            let decompressor = GDeflateDecompressor::new();

            let base_in_ptr = tile_offsets.add(header.num_tiles as usize) as *const u8;

            for tile_index in 0..header.num_tiles {
                let tile_offset = if tile_index > 0 {
                    *tile_offsets.add(tile_index as usize)
                } else {
                    0
                } as usize;

                let data_size = if tile_index < header.num_tiles - 1 {
                    *tile_offsets.add(tile_index as usize + 1) - tile_offset as u32
                } else {
                    *tile_offsets
                };
                let data_ptr = base_in_ptr.add(tile_offset) as *const std::ffi::c_void;

                let mut compressed_page = libdeflate_gdeflate_in_page {
                    data: data_ptr,
                    nbytes: data_size as usize,
                };

                let output_offset = tile_index as usize * KDEFAULT_TILE_SIZE;
                let out_ptr =
                    out_data.as_mut_ptr().add(output_offset) as *mut std::ffi::c_void;
                let mut out_nbytes = 0;
                let decomp_result: libdeflate_result = libdeflate_gdeflate_decompress(
                    decompressor.p,
                    &mut compressed_page,
                    1,
                    out_ptr,
                    KDEFAULT_TILE_SIZE,
                    &mut out_nbytes,
                ) as libdeflate_result;
                
                let _res = match decomp_result {
                    libdeflate_result_LIBDEFLATE_SUCCESS => bytes_read += out_nbytes,
                    libdeflate_result_LIBDEFLATE_BAD_DATA => {
                        return Err(DecompressionError::BadData)
                    },
                    libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                        return Err(DecompressionError::InsufficientSpace)
                    }
                    _ => {
                        panic!("libdeflate_gdeflate_decompress returned an unknown error type: this is an internal bug that **must** be fixed");
                    }
                };
            }
        }
        Ok(bytes_read)
    }
}

#[no_mangle]
pub extern "C" fn gdeflate_decompress(
    out_data: *mut u8,
    out_size: u64,
    in_data: *const u8,
    in_size: u64,
    num_workers: u32,
) -> bool {
    if in_data.is_null() || out_data.is_null() {
        return false;
    }
    let input = unsafe { std::slice::from_raw_parts(in_data, in_size as usize)};
    let output = unsafe { std::slice::from_raw_parts_mut(out_data, out_size as usize)};
    if num_workers == 0 {
        return match GDeflateDecompressor::gdeflate_decompress(input, output) {
            Ok(_size) => true,
            Err(_) => false,
        }
    }
    else {
        return match GDeflateDecompressor::gdeflate_decompress_threaded(input, output, num_workers) {
            Ok(_size) => true,
            Err(_) => false,
        }
    }

}

#[no_mangle]
pub extern "C" fn gdeflate_get_uncompressed_size(
    in_data: *const u8,
    in_size: u64,
    uncompressed_size: *mut u64
) -> bool {
    if in_data.is_null() {
        return false;
    }
    let input = unsafe { std::slice::from_raw_parts(in_data, in_size as usize)};
    let mut cursor = Cursor::new(input);
    let stream = match TileStream::from(&mut cursor) {
        Ok(x) => x,
        Err(_e) => return false,
    };
    unsafe {
        *uncompressed_size = stream.get_uncompressed_size() as u64;
    }
    true

}

#[no_mangle]
pub extern "C" fn gdeflate_compress(
    out_data: *mut u8,
    out_size: *mut u64,
    in_data: *const u8,
    in_size: u64,
    level: u32,
    flags: u32,
) -> bool {
    return false;
}


/// A `libdeflate` decompressor that can inflate DEFLATE, zlib, or
/// gzip data.
pub struct Decompressor {
    p: *mut libdeflate_decompressor,
}
unsafe impl Send for Decompressor {}

/// An error that may be returned by one of the
/// [`Decompressor`](struct.Decompressor.html)'s `decompress_*`
/// methods when a decompression cannot be performed.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DecompressionError {
    /// The provided data is invalid in some way. For example, the
    /// checksum in the data revealed possible corruption, magic
    /// numbers in the data do not match expectations, etc.
    BadData,

    /// The provided output buffer is not large enough to accomodate
    /// the decompressed data.
    InsufficientSpace,
}

impl fmt::Display for DecompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            DecompressionError::BadData => write!(f, "the data provided to a libdeflater *_decompress function call was invalid in some way (e.g. bad magic numbers, bad checksum)"),
            DecompressionError::InsufficientSpace => write!(f, "a buffer provided to a libdeflater *_decompress function call was too small to accommodate the decompressed data")
        }
    }
}

impl Error for DecompressionError {}

/// A result returned by decompression methods
type DecompressionResult<T> = std::result::Result<T, DecompressionError>;

#[allow(non_upper_case_globals)]
impl Decompressor {
    /// Returns a newly constructed instance of a `Decompressor`.
    pub fn new() -> Decompressor {
        unsafe {
            init_allocator();
            let ptr = libdeflate_alloc_decompressor();
            if !ptr.is_null() {
                Decompressor { p: ptr }
            } else {
                panic!("libdeflate_alloc_decompressor returned NULL: out of memory");
            }
        }
    }

    /// Decompresses `gz_data` (a buffer containing
    /// [`gzip`](https://tools.ietf.org/html/rfc1952) data) and writes
    /// the decompressed data into `out`. Returns the number of
    /// decompressed bytes written into `out`, or an error (see
    /// [`DecompressionError`](enum.DecompressionError.html) for error
    /// cases).
    pub fn gzip_decompress(
        &mut self,
        gz_data: &[u8],
        out: &mut [u8],
    ) -> DecompressionResult<usize> {
        unsafe {
            init_allocator();
            let mut out_nbytes = 0;
            let in_ptr = gz_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out.as_mut_ptr() as *mut std::ffi::c_void;
            let ret: libdeflate_result = libdeflate_gzip_decompress(
                self.p,
                in_ptr,
                gz_data.len(),
                out_ptr,
                out.len(),
                &mut out_nbytes,
            );
            match ret {
                libdeflate_result_LIBDEFLATE_SUCCESS => Ok(out_nbytes),
                libdeflate_result_LIBDEFLATE_BAD_DATA => Err(DecompressionError::BadData),
                libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                    Err(DecompressionError::InsufficientSpace)
                }
                _ => {
                    panic!("libdeflate_gzip_decompress returned an unknown error type: this is an internal bug that **must** be fixed");
                }
            }
        }
    }

    /// Decompresses `zlib_data` (a buffer containing
    /// [`zlib`](https://www.ietf.org/rfc/rfc1950.txt) data) and
    /// writes the decompressed data to `out`. Returns the number of
    /// decompressed bytes written into `out`, or an error (see
    /// [`DecompressionError`](enum.DecompressionError.html) for error
    /// cases).
    pub fn zlib_decompress(
        &mut self,
        zlib_data: &[u8],
        out: &mut [u8],
    ) -> DecompressionResult<usize> {
        unsafe {
            init_allocator();
            let mut out_nbytes = 0;
            let in_ptr = zlib_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out.as_mut_ptr() as *mut std::ffi::c_void;
            let ret: libdeflate_result = libdeflate_zlib_decompress(
                self.p,
                in_ptr,
                zlib_data.len(),
                out_ptr,
                out.len(),
                &mut out_nbytes,
            );

            match ret {
                libdeflate_result_LIBDEFLATE_SUCCESS => Ok(out_nbytes),
                libdeflate_result_LIBDEFLATE_BAD_DATA => Err(DecompressionError::BadData),
                libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                    Err(DecompressionError::InsufficientSpace)
                }
                _ => {
                    panic!("libdeflate_zlib_decompress returned an unknown error type: this is an internal bug that **must** be fixed");
                }
            }
        }
    }

    /// Decompresses `deflate_data` (a buffer containing
    /// [`deflate`](https://tools.ietf.org/html/rfc1951) data) and
    /// writes the decompressed data to `out`. Returns the number of
    /// decompressed bytes written into `out`, or an error (see
    /// [`DecompressionError`](enum.DecompressionError.html) for error
    /// cases).
    pub fn deflate_decompress(
        &mut self,
        deflate_data: &[u8],
        out: &mut [u8],
    ) -> DecompressionResult<usize> {
        unsafe {
            init_allocator();
            let mut out_nbytes = 0;
            let in_ptr = deflate_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out.as_mut_ptr() as *mut std::ffi::c_void;
            let ret: libdeflate_result = libdeflate_deflate_decompress(
                self.p,
                in_ptr,
                deflate_data.len(),
                out_ptr,
                out.len(),
                &mut out_nbytes,
            );

            match ret {
                libdeflate_result_LIBDEFLATE_SUCCESS => Ok(out_nbytes),
                libdeflate_result_LIBDEFLATE_BAD_DATA => Err(DecompressionError::BadData),
                libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                    Err(DecompressionError::InsufficientSpace)
                }
                _ => {
                    panic!("libdeflate_deflate_decompress returned an unknown error type: this is an internal bug that **must** be fixed");
                }
            }
        }
    }
}

impl Drop for Decompressor {
    fn drop(&mut self) {
        unsafe {
            libdeflate_free_decompressor(self.p);
        }
    }
}

impl Drop for GDeflateDecompressor {
    fn drop(&mut self) {
        unsafe {
            libdeflate_free_gdeflate_decompressor(self.p);
        }
    }
}

/// Raw numeric values of compression levels that are accepted by libdeflate
const MIN_COMPRESSION_LVL: i32 = 0;
const DEFAULT_COMPRESSION_LVL: i32 = 6;
const MAX_COMPRESSION_LVL: i32 = 12;

/// Compression level used by a [`Compressor`](struct.Compressor.html)
/// instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CompressionLvl(i32);

/// Errors that can be returned when attempting to create a
/// [`CompressionLvl`](enum.CompressionLvl.html) from a numeric value.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CompressionLvlError {
    InvalidValue,
}

/// A result that is returned when trying to create a
/// [`CompressionLvl`](enum.CompressionLvl.html) from a numeric value.
type CompressionLevelResult = Result<CompressionLvl, CompressionLvlError>;

impl CompressionLvl {
    /// Try to create a valid
    /// [`CompressionLvl`](enum.CompressionLvl.html) from a numeric
    /// value.
    ///
    /// If `level` is a valid custom compression level for libdeflate,
    /// returns a `Result::Ok(CompressionLvl)`. Otherwise, returns
    /// `Result::Error(error)`.
    ///
    /// Valid compression levels for libdeflate, at time of writing,
    /// are 1-12.
    pub fn new(level: i32) -> CompressionLevelResult {
        if MIN_COMPRESSION_LVL <= level && level <= MAX_COMPRESSION_LVL {
            Ok(CompressionLvl(level))
        } else {
            Err(CompressionLvlError::InvalidValue)
        }
    }

    /// Returns the fastest compression level. This compression level
    /// offers the highest performance but lowest compression ratio.
    pub fn fastest() -> CompressionLvl {
        CompressionLvl(MIN_COMPRESSION_LVL)
    }

    /// Returns the best compression level, in terms of compression
    /// ratio. This compression level offers the best compression
    /// ratio but lowest performance.
    pub fn best() -> CompressionLvl {
        CompressionLvl(MAX_COMPRESSION_LVL)
    }

    /// Returns an iterator that emits all compression levels
    /// supported by `libdeflate` in ascending order.
    pub fn iter() -> CompressionLvlIter {
        CompressionLvlIter(MIN_COMPRESSION_LVL)
    }
}

impl Default for CompressionLvl {
    /// Returns the default compression level reccomended by
    /// libdeflate.
    fn default() -> CompressionLvl {
        CompressionLvl(DEFAULT_COMPRESSION_LVL)
    }
}

/// An iterator over the
/// [`CompressionLvl`](struct.CompressionLvl.html)s supported by the
/// [`Compressor`](struct.Compressor.html).
pub struct CompressionLvlIter(i32);

impl Iterator for CompressionLvlIter {
    type Item = CompressionLvl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 <= MAX_COMPRESSION_LVL {
            let ret = Some(CompressionLvl(self.0));
            self.0 += 1;
            ret
        } else {
            None
        }
    }
}

impl From<CompressionLvl> for i32 {
    fn from(level: CompressionLvl) -> Self {
        level.0
    }
}

impl From<&CompressionLvl> for i32 {
    fn from(level: &CompressionLvl) -> Self {
        level.0
    }
}

/// An error that may be returned when calling one of the
/// [`Compressor`](struct.Compressor.html)'s `compress_*` methods.
#[derive(Debug, PartialEq)]
pub enum CompressionError {
    InsufficientSpace,
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            CompressionError::InsufficientSpace => write!(f, "the output buffer provided to a libdeflater *_compress function call was too small for the input data"),
        }
    }
}

impl Error for CompressionError {}

type CompressionResult<T> = std::result::Result<T, CompressionError>;

/// A `libdeflate` compressor that can compress arbitrary data into
/// DEFLATE, zlib, or gzip formats.
pub struct Compressor {
    p: *mut libdeflate_compressor,
}
unsafe impl Send for Compressor {}

impl Compressor {
    /// Returns a newly constructed `Compressor` that compresses data
    /// with the supplied
    /// [`CompressionLvl`](struct.CompressionLvl.html)
    pub fn new(lvl: CompressionLvl) -> Compressor {
        unsafe {
            init_allocator();
            let ptr = libdeflate_alloc_compressor(lvl.0);
            if !ptr.is_null() {
                Compressor { p: ptr }
            } else {
                panic!("libdeflate_alloc_compressor returned NULL: out of memory");
            }
        }
    }

    /// Returns the maximum number of bytes required to encode
    /// `n_bytes` as [`deflate`](https://tools.ietf.org/html/rfc1951)
    /// data. This is a hard upper-bound that assumes the worst
    /// possible compression ratio (i.e. assumes the data cannot be
    /// compressed), format overhead, etc.
    pub fn deflate_compress_bound(&mut self, n_bytes: usize) -> usize {
        unsafe { libdeflate_deflate_compress_bound(self.p, n_bytes) }
    }

    /// Compresses `in_raw_data` as
    /// [`deflate`](https://tools.ietf.org/html/rfc1951) data, writing
    /// the data into `out_deflate_data`. Returns the number of bytes
    /// written into `out_deflate_data`.
    pub fn deflate_compress(
        &mut self,
        in_raw_data: &[u8],
        out_deflate_data: &mut [u8],
    ) -> CompressionResult<usize> {
        unsafe {
            init_allocator();
            let in_ptr = in_raw_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out_deflate_data.as_mut_ptr() as *mut std::ffi::c_void;

            let sz = libdeflate_deflate_compress(
                self.p,
                in_ptr,
                in_raw_data.len(),
                out_ptr,
                out_deflate_data.len(),
            );

            if sz != 0 {
                Ok(sz)
            } else {
                Err(CompressionError::InsufficientSpace)
            }
        }
    }

    /// Returns the maximum number of bytes required to encode
    /// `n_bytes` as [`zlib`](https://www.ietf.org/rfc/rfc1950.txt)
    /// data. This is a hard upper-bound that assumes the worst
    /// possible compression ratio (i.e. assumes the data cannot be
    /// compressed), format overhead, etc.
    pub fn zlib_compress_bound(&mut self, n_bytes: usize) -> usize {
        unsafe { libdeflate_zlib_compress_bound(self.p, n_bytes) }
    }

    /// Compresses `in_raw_data` as
    /// [`zlib`](https://www.ietf.org/rfc/rfc1950.txt) data, writing
    /// the data into `out_zlib_data`. Returns the number of bytes
    /// written into `out_zlib_data`.
    pub fn zlib_compress(
        &mut self,
        in_raw_data: &[u8],
        out_zlib_data: &mut [u8],
    ) -> CompressionResult<usize> {
        unsafe {
            init_allocator();
            let in_ptr = in_raw_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out_zlib_data.as_mut_ptr() as *mut std::ffi::c_void;

            let sz = libdeflate_zlib_compress(
                self.p,
                in_ptr,
                in_raw_data.len(),
                out_ptr,
                out_zlib_data.len(),
            );

            if sz != 0 {
                Ok(sz)
            } else {
                Err(CompressionError::InsufficientSpace)
            }
        }
    }

    /// Returns the maximum number of bytes required to encode
    /// `n_bytes` as [`gzip`](https://tools.ietf.org/html/rfc1952)
    /// data. This is a hard upper-bound that assumes the worst
    /// possible compression ratio (i.e. assumes the data cannot be
    /// compressed), format overhead, etc.
    pub fn gzip_compress_bound(&mut self, n_bytes: usize) -> usize {
        unsafe { libdeflate_gzip_compress_bound(self.p, n_bytes) }
    }

    /// Compresses `in_raw_data` as
    /// [`gzip`](https://tools.ietf.org/html/rfc1952) data, writing
    /// the data into `out_gzip_data`. Returns the number of bytes
    /// written into `out_gzip_data`.
    pub fn gzip_compress(
        &mut self,
        in_raw_data: &[u8],
        out_gzip_data: &mut [u8],
    ) -> CompressionResult<usize> {
        unsafe {
            init_allocator();
            let in_ptr = in_raw_data.as_ptr() as *const std::ffi::c_void;
            let out_ptr = out_gzip_data.as_mut_ptr() as *mut std::ffi::c_void;

            let sz = libdeflate_gzip_compress(
                self.p,
                in_ptr,
                in_raw_data.len(),
                out_ptr,
                out_gzip_data.len(),
            );

            if sz != 0 {
                Ok(sz)
            } else {
                Err(CompressionError::InsufficientSpace)
            }
        }
    }
}

impl Drop for Compressor {
    fn drop(&mut self) {
        unsafe {
            libdeflate_free_compressor(self.p);
        }
    }
}

/// Struct holding the state required to compute a rolling crc32
/// value.
pub struct Crc {
    val: u32,
}

impl Crc {
    /// Returns a new `Crc` instance
    pub fn new() -> Crc {
        Crc { val: 0 }
    }

    /// Update the CRC with the bytes in `data`
    pub fn update(&mut self, data: &[u8]) {
        unsafe {
            self.val = libdeflate_crc32(
                self.val,
                data.as_ptr() as *const core::ffi::c_void,
                data.len(),
            );
        }
    }

    /// Returns the current CRC32 checksum
    pub fn sum(&self) -> u32 {
        self.val
    }
}

/// Returns the CRC32 checksum of the bytes in `data`.
///
/// Note: this is a one-shot method that requires all data
/// up-front. Developers wanting to compute a rolling crc32 from
/// (e.g.) a stream should use [`Crc`](struct.Crc.html)
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = Crc::new();
    crc.update(&data);
    crc.sum()
}

/// Struct holding the state required to compute a rolling adler32
/// value.
pub struct Adler32 {
    val: u32,
}

impl Adler32 {
    /// Returns a new `Adler32` instance (with initial adler32 value 1, which is default for adler32)
    pub fn new() -> Adler32 {
        Adler32 { val: 1 }
    }
    /// Update the Adler32 with the bytes in `data`
    pub fn update(&mut self, data: &[u8]) {
        unsafe {
            self.val = libdeflate_adler32(
                self.val,
                data.as_ptr() as *const core::ffi::c_void,
                data.len(),
            );
        }
    }
    /// Returns the current Adler32 checksum
    pub fn sum(&self) -> u32 {
        self.val
    }
}
/// Returns the Adler32 checksum of the bytes in `data`.
///
/// Note: this is a one-shot method that requires all data
/// up-front. Developers wanting to compute a rolling adler32 from
/// (e.g.) a stream should use [`Adler32`](struct.Adler32.html)
pub fn adler32(data: &[u8]) -> u32 {
    let mut adler32 = Adler32::new();
    adler32.update(&data);
    adler32.sum()
}

