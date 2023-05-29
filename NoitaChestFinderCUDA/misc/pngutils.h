#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../Configuration.h"

#include <stdlib.h>

#define PNG_DEBUG 3
#include <png.h>

void WriteImage(const char* file_name, uint8_t* data, int w, int h)
{
	png_bytep* rows = (png_bytep*)malloc(sizeof(void*) * h);
	for (int y = 0; y < h; y++)
	{
		rows[y] = data + 3 * y * w;
	}

	/* create file */
	FILE *fp = fopen(file_name, "wb");

	png_structp png_ptr;
	png_infop info_ptr;

	png_byte color_type = PNG_COLOR_TYPE_RGB;
	png_byte bit_depth = 8;

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	info_ptr = png_create_info_struct(png_ptr);

	png_init_io(png_ptr, fp);

	/* write header */

	png_set_IHDR(png_ptr, info_ptr, w, h,
					bit_depth, color_type, PNG_INTERLACE_NONE,
					PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	/* write bytes */
	png_write_image(png_ptr, rows);

	/* end write */

	png_write_end(png_ptr, NULL);

	fclose(fp);

	free(rows);
}

IntPair GetImageDimensions(const char* file_name)
{
	char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE* fp = fopen(file_name, "rb");
	fread(header, 1, 8, fp);

	/* initialize stuff */
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	png_infop info_ptr = png_create_info_struct(png_ptr);

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	int w = png_get_image_width(png_ptr, info_ptr);
	int h = png_get_image_height(png_ptr, info_ptr);

	fclose(fp);

	return { w, h };
}

void ReadImage(const char* file_name, uint8_t* data)
{
	char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE* fp = fopen(file_name, "rb");
	fread(header, 1, 8, fp);

	/* initialize stuff */
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	png_infop info_ptr = png_create_info_struct(png_ptr);

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	int w = png_get_image_width(png_ptr, info_ptr);
	int h = png_get_image_height(png_ptr, info_ptr);
	png_byte color_type = png_get_color_type(png_ptr, info_ptr);
	png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	int number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);


	/* read file */
	png_bytepp rows = (png_bytepp)malloc(sizeof(png_bytep) * h);
	for (int y = 0; y < h; y++)
	{
		rows[y] = data + 3 * y * w;
	}

	/* read file */
	png_read_image(png_ptr, rows);

	png_read_end(png_ptr, info_ptr);

	fclose(fp);

	free(rows);
}

void ConvertRGBAToRGB(const char* file_name)
{
	char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE* fp = fopen(file_name, "rb");
	fread(header, 1, 8, fp);

	/* initialize stuff */
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	png_infop info_ptr = png_create_info_struct(png_ptr);

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	int w = png_get_image_width(png_ptr, info_ptr);
	int h = png_get_image_height(png_ptr, info_ptr);
	png_byte color_type = png_get_color_type(png_ptr, info_ptr);
	png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	int number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);

	uint8_t* dat1 = (uint8_t*)malloc(4 * w * h);
	uint8_t* dat2 = (uint8_t*)malloc(4 * w * h);

	/* read file */
	png_bytepp rows = (png_bytepp)malloc(sizeof(png_bytep) * h);
	for (int y = 0; y < h; y++)
	{
		rows[y] = dat1 + 4 * y * w;
	}

	/* read file */
	png_read_image(png_ptr, rows);

	fclose(fp);

	free(rows);

	for (int i = 0; i < w * h; i++)
	{
		memcpy(dat2 + 3 * i, dat1 + 4 * i, 3);
	}
	WriteImage(file_name, dat2, w, h);
}