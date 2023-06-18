#include "HDF5.h"

template <>
inline H5::DataType HDF::GetH5_Type<bool>(bool __attribute__((unused)) const val) const
{ return H5::PredType::NATIVE_HBOOL; }

template <>
inline H5::DataType HDF::GetH5_Type<int>(int __attribute__((unused)) const val) const
{ return H5::PredType::NATIVE_UINT32; }

template <>
inline H5::DataType HDF::GetH5_Type<uint32_t>(uint32_t __attribute__((unused)) const val) const
{ return H5::PredType::NATIVE_UINT32; }

template <>
inline H5::DataType HDF::GetH5_Type<uint64_t>(uint64_t __attribute__((unused)) const val) const
{ return H5::PredType::NATIVE_UINT64; }

template <>
inline H5::DataType HDF::GetH5_Type<double>(double __attribute__((unused)) const val) const
{ return H5::PredType::NATIVE_DOUBLE; }

HDF::HDF(std::string const& fileName, bool const read)
	: file{[read, fileName]() {
		if (read == false)
			return H5::H5File(fileName, H5F_ACC_TRUNC);
		auto tmp = std::ofstream(fileName);		// create the file, if it doesn't exist
		return H5::H5File(fileName, H5F_ACC_RDWR);
		}()}
	, fileName{fileName}
{ }

HDF::HDF()
	: file{}
	, fileName{}
{ }

void HDF::CreateGroup(std::string const& groupName) {
	if (file.exists(groupName.c_str()) == false)
		file.createGroup(groupName.c_str());
}

void HDF::ReadSeeds(std::string const& groupName, std::string const& name, ManagedVector<curandState>& data) const {
	auto const type = H5::PredType::NATIVE_OPAQUE;
	auto const grp = file.openGroup(groupName.c_str());
	auto const Set = grp.openDataSet(name.c_str());
	Set.read(data.data(), type);
}

void HDF::WriteSeeds(std::string const& groupName, std::string const& name, ManagedVector<curandState> const& data) {
	auto const type = H5::PredType::NATIVE_OPAQUE;
	hsize_t h5sizes[] = {data.size() * sizeof(curandState)};
	auto ds = H5::DataSpace{1, h5sizes};
	auto grp = file.openGroup(groupName.c_str());
	auto Set = (grp.exists(name.c_str()) == true ? grp.openDataSet(name.c_str()) : 
			grp.createDataSet(name.c_str(), type, ds));
	Set.write(data.data(), type);
}

int HDF::NumberOfConfigs() const {
	if (file.exists("/data")) {
		auto const grp = file.openGroup("/data/");
		return grp.getNumObjs();
	} else {
		return -1;
	}
}

std::string HDF::NameOfConfig(int const cnfg) const {
	auto const grp = file.openGroup("/data/");
	return grp.getObjnameByIdx(cnfg);
}

std::string HDF::NameOfLastConfig() const {
	auto const grp = file.openGroup("/data/");
	return grp.getObjnameByIdx(grp.getNumObjs()-1);
}

template <typename T>
void HDF::WriteData(std::string const& groupName, std::string const& name, T const& data) {
	auto const type = H5::PredType::NATIVE_DOUBLE;	//TODO: fixed to double for the time being!!!!!!!!
	// data is stored as (n, x_0, x_1, ..., x_{d-1}), where 'n' stands for the field components
	hsize_t h5sizes[nDim + 1];
	for (int i = 0; i < nDim; ++i)
		h5sizes[i+1] = Sizes[i];
	h5sizes[0] = nVectorComponents;
	auto ds = H5::DataSpace{nDim+1, h5sizes};
	auto grp = file.openGroup(groupName.c_str());
	auto Set = (grp.exists(name.c_str()) == true ? grp.openDataSet(name.c_str()) : 
			grp.createDataSet(name.c_str(), type, ds));
	Set.write(data.data(), type);
}

template <typename T>
void HDF::ReadData(std::string const& groupName, std::string const& name, T& data) const {
	auto const type = H5::PredType::NATIVE_DOUBLE;	//TODO: fixed to double for the time being!!!!!!!!
	auto const grp = file.openGroup(groupName.c_str());
	auto const Set = grp.openDataSet(name.c_str());
	if (Set.getStorageSize() != SIZE * nVectorComponents * sizeof(myType)) {
		std::cerr << "Expected " << SIZE << " sites, but found "
			<< Set.getStorageSize() / (nVectorComponents * sizeof(myType)) <<" instead.\nExiting.\n";
		exit(1);
	}
	Set.read(data.data(), type);
}

template <typename T>
void HDF::UpdateH5_Attribute(std::string const& groupName, std::string const& attributeName, T const val) {
	auto grp = file.openGroup(groupName.c_str());
	auto const type = GetH5_Type(val);
	auto attr = grp.openAttribute(attributeName.c_str(), type, H5::DataSpace());
	attr.write(type, &val);
}

template <typename T>
void HDF::WriteH5_Attribute(std::string const& groupName, std::string const& attributeName, T const val) {
	auto grp = file.openGroup(groupName.c_str());
	auto const type = GetH5_Type(val);
	auto attr = grp.createAttribute(attributeName.c_str(), type, H5::DataSpace());
	attr.write(type, &val);
}

template <typename T>
T HDF::ReadH5_Attribute(std::string const& groupName, std::string const& attributeName) const {
	auto val = T{};
	auto const grp = file.openGroup(groupName.c_str());
	auto const type = GetH5_Type(val);
	auto attr = grp.openAttribute(attributeName.c_str());
	attr.read(type, &val);
	return val;
}


template void HDF::WriteH5_Attribute<int>(std::string const& groupName, std::string const& attributeName, int const val);
template void HDF::WriteH5_Attribute<uint32_t>(std::string const& groupName, std::string const& attributeName, uint32_t const val);
template void HDF::WriteH5_Attribute<uint64_t>(std::string const& groupName, std::string const& attributeName, uint64_t const val);
template void HDF::WriteH5_Attribute<double>(std::string const& groupName, std::string const& attributeName, double const val);

template int HDF::ReadH5_Attribute<int>(std::string const& groupName, std::string const& attributeName) const;
template uint32_t HDF::ReadH5_Attribute<uint32_t>(std::string const& groupName, std::string const& attributeName) const;
template uint64_t HDF::ReadH5_Attribute<uint64_t>(std::string const& groupName, std::string const& attributeName) const;
template double HDF::ReadH5_Attribute<double>(std::string const& groupName, std::string const& attributeName) const;

template void HDF::WriteData(std::string const& groupName, std::string const& name, std::vector<myType> const& data);
template void HDF::WriteData(std::string const& groupName, std::string const& name, ManagedVector<myType> const& data);

template void HDF::ReadData(std::string const& groupName, std::string const& name, std::vector<myType>& data) const;
template void HDF::ReadData(std::string const& groupName, std::string const& name, ManagedVector<myType>& data) const;
