#pragma once
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string_view>

void read_wak(const char* wak_path);
std::string& get_wak_file(const std::string& path);
std::string find_wak();