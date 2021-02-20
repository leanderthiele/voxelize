#ifndef READ_HDF5_HPP
#define READ_HDF5_HPP

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

#include "H5Cpp.h"
#include "H5File.h"
#include "H5public.h"

namespace ReadHDF5 {

static void *
read_field (std::shared_ptr<H5::H5File> fptr, const std::string &name,
            size_t element_size, size_t Nitems, size_t dim)
{
    auto dset = fptr->openDataSet(name);
    auto dspace = dset.getSpace();

    hsize_t dim_lengths[16];
    auto Ndims = dspace.getSimpleExtentDims(dim_lengths);
    auto Dtype = dset.getDataType();

    assert(Dtype.getSize() == element_size);
    assert(Nitems == dim_lengths[0]);
    assert((Ndims==1 && dim==1) || (Ndims==2 && dim_lengths[1]==dim));
    assert(dset.getInMemDataSize() == Nitems * dim * element_size);

    void *data = std::malloc(dset.getInMemDataSize());
    auto memspace = H5::DataSpace(Ndims, dim_lengths);
    dset.read(data, Dtype, memspace, dspace);

    return data;
}

template<typename TH5, typename Trequ>
static Trequ
read_header_attr_scalar (std::shared_ptr<H5::H5File> fptr, const std::string &name)
{
    TH5 out;
    auto header = fptr->openGroup("/Header");
    auto attr = header.openAttribute(name);
    assert(attr.getDataType().getSize() == sizeof(TH5));
    attr.read(attr.getDataType(), &out);
    return (Trequ)(out);
}

template<typename TH5, typename Trequ>
static Trequ
read_header_attr_vector (std::shared_ptr<H5::H5File> fptr, const std::string &name, size_t idx)
{
    auto header = fptr->openGroup("/Header");
    auto attr = header.openAttribute(name);
    auto aspace = attr.getSpace();
    hsize_t dim_lengths[16];
    auto Ndims = aspace.getSimpleExtentDims(dim_lengths);
    assert(Ndims==1);
    assert(dim_lengths[0] > idx);
    assert(attr.getDataType().getSize() == sizeof(TH5));
    TH5 out[dim_lengths[0]];
    attr.read(attr.getDataType(), out);
    return (Trequ)(out[idx]);
}

} // namespace ReadHDF5

#endif // READ_HDF5_HPP
