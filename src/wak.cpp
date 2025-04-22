#pragma once
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WIN32
#define NOMINMAX
#include "Windows.h"
#else
#include <iostream>
#endif

// Portable solution probably doesn't exist, sucks to suck
static std::string locate_file_dialog(const char* hint, const char* filter, const char* title) {
#ifdef WIN32
	char buf[2048] = {};
	OPENFILENAMEA fn = {};
	fn.lStructSize = sizeof(OPENFILENAMEA);
	fn.lpstrFilter = (LPSTR)filter;
	fn.lpstrFile = (LPSTR)buf;
	fn.nMaxFile = 2048;
	fn.lpstrTitle = (LPSTR)title;
	fn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER;
	GetOpenFileNameA(&fn);
	return std::string(buf);
#else
	// yep, it's terrible. Linux people, make this better if possible maybe
	char buf[1024];
	printf("%s\nThis would be a dialog box, but you're on the wrong operating system. Enter the path here:\n>", title);
	std::cin.getline(buf, 1024);
	return buf;
#endif
}

std::unordered_map<std::string, std::string> globalWakContents;

template <typename T>
static T read_le(std::istream&);
template <>
std::uint8_t read_le(std::istream& s) {
	uint8_t val;
	s.read((char*)&val, sizeof(val));
	return val;
}
template <>
std::uint32_t read_le(std::istream& s) {
	uint32_t val;
	auto it = (uint8_t*)&val;
	for (int i = 0; i < 4; i++)
		it[i] = read_le<uint8_t>(s);
	return val;
}
template <>
std::string read_le(std::istream& s) {
	std::uint32_t size = read_le<std::uint32_t>(s);
	std::string str;
	str.resize(size);
	s.read((char*)str.data(), size);
	return str;
}

static std::string read_file(const char* path) {
	std::string out;
	std::ifstream stream(path, std::ios::binary);
	if (stream.fail()) {
		printf("[%s] does not exist.\n", path);
		exit(-1);
	}
	while (stream) {
		char buffer[1024];
		stream.read(buffer, sizeof(buffer));
		out.append(buffer, stream.gcount());
	}

	return out;
}
static void write_file(const char* path, const std::string& in) {
	std::ofstream stream(path, std::ios::binary);
	stream.write(in.c_str(), in.length());
}

void read_wak(const char* wak_path) {
	globalWakContents.clear();
	std::string contents = read_file(wak_path);
	std::istringstream data(contents);

	uint32_t z1 = read_le<std::uint32_t>(data);
	uint32_t fileCount = read_le<std::uint32_t>(data);
	uint32_t dataStart = read_le<std::uint32_t>(data);
	uint32_t z2 = read_le<std::uint32_t>(data);

	for (uint32_t i = 0; i < fileCount; i++) {
		uint32_t offset = read_le<std::uint32_t>(data);
		uint32_t size = read_le<std::uint32_t>(data);
		std::string name = read_le<std::string>(data);
		globalWakContents.emplace(name, std::string(contents.c_str() + offset, size));
	}
}

std::string& get_wak_file(const std::string& path) {
	if (globalWakContents.find(path) != globalWakContents.end()) {
		return globalWakContents.at(path);
	} else {
		fprintf(stderr, "Unable to find file: %s\n", path.c_str());
		exit(-1);
	}
}

std::string find_wak() {
#ifndef __CUDA_ARCH__
	const char* wakpath = ".wakpath";
	if (std::filesystem::exists(wakpath))
		return read_file(wakpath);

	std::filesystem::path current = std::filesystem::current_path();
	std::filesystem::path dialog_ret = locate_file_dialog(
		"", "data.wak\0data.wak\0", "Locate your install's data.wak, in the steam install directory");
	std::filesystem::current_path(current);
	write_file(wakpath, dialog_ret.string().c_str());

	if (dialog_ret.filename().string() != "data.wak")
		exit(-1);

	return dialog_ret.string();
#else
	return "";
#endif
}