#pragma once

#include <string>
#include <vector>
#include <H5Cpp.h>
#include <fstream>
#include <iostream>
#include <curand_kernel.h>

#include "params.h"
#include "managedVector.h"

class HDF {
	H5::H5File file;
	std::string const fileName;

	template <typename T>
	H5::DataType GetH5_Type(T __attribute__((unused)) const val) const;

public:
	HDF(std::string const& fileName, bool const read);
	HDF();

	inline void Open() { file.openFile(fileName, H5F_ACC_RDWR); }
	inline void Close() { file.close(); }

	template <typename T>
	void UpdateH5_Attribute(std::string const& groupName, std::string const& attributeName, T const val);
	template <typename T>
	void WriteH5_Attribute(std::string const& groupName, std::string const& attributeName, T const val);
	template <typename T>
	T ReadH5_Attribute(std::string const& groupName, std::string const& attributeName) const;

	template <typename T>
	void WriteData(std::string const& groupName, std::string const& name, T const& data);
	template <typename T>
	void ReadData(std::string const& groupName, std::string const& name, T& data) const;

	void CreateGroup(std::string const& groupName);
	void ReadSeeds(std::string const& groupName, std::string const& name, ManagedVector<curandState>& data) const;
	void WriteSeeds(std::string const& groupName, std::string const& name, ManagedVector<curandState> const& data);

	int NumberOfConfigs() const;
	std::string NameOfLastConfig() const;
	std::string NameOfConfig(int const cnfg) const;
};
