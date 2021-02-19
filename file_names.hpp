#ifndef FILE_NAMES_HPP
#define FILE_NAMES_HPP

#include <cstdio>
#include <utility>
#include <string>
#include <stdexcept>
#include <type_traits>

template<typename T>
inline bool
extract_number (const std::string &fname,
                std::pair<const std::string, const std::string> bounds,
                T &out)
{// {{{
    // extracts anything between bounds.first and bounds.second and tries to convert to float,
    // write into out

    size_t pos1 = fname.find(bounds.first);
    if (pos1 == std::string::npos)
        return false;
    pos1 += bounds.first.size();

    size_t pos2 = fname.find(bounds.second, pos1);
    if (pos2 == std::string::npos)
        return false;
    
    auto sub = fname.substr(pos1, pos2-pos1);

    try
    {
        if constexpr (std::is_same_v<T,float>)
            out = std::stof(sub);
        else if constexpr (std::is_same_v<T,double>)
            out = std::stod(sub);
        else if constexpr (std::is_same_v<T,long double>)
            out = std::stold(sub);
        else if constexpr (std::is_same_v<T,int>)
            out = std::stoi(sub);
        else if constexpr (std::is_same_v<T,long>)
            out = std::stol(sub);
        else if constexpr (std::is_same_v<T,unsigned long>)
            out = std::stoul(sub);
        else
            return false;
    }
    catch (std::invalid_argument &e)
    {
        std::fprintf(stderr, "File name %s does not conform to expectation, "
                             "could not extract limit %s.\n",
                             fname.c_str(), bounds.first.c_str());
    }

    return true;
}// }}}

std::string get_fname (const std::string &prefix, float Rmin, float Rmax, const std::string &suffix = "")
{// {{{
    char buffer[512];
    std::sprintf(buffer, "%s_Rmin%.8e_Rmax%.8e_%s.pt", prefix.c_str(), Rmin, Rmax, suffix.c_str());
    return std::string(buffer);
}// }}}

bool read_fname (const std::string &fname, float &Rmin, float &Rmax)
{// {{{
    if (!extract_number(fname, std::make_pair("Rmin", "_"), Rmin))
        return false;
    if (!extract_number(fname, std::make_pair("Rmax", "_"), Rmax))
        return false;
    return true;
}// }}}

#endif
