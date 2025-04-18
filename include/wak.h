#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

void read_wak(const char* wak_path);
std::string& get_wak_file(const std::string& path);
std::string find_wak();