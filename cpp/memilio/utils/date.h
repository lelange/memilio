/* 
* Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
*
* Authors: Daniel Abele, Martin J. Kuehn
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef EPI_UTILS_DATE_H
#define EPI_UTILS_DATE_H

#include "memilio/io/io.h"
#include <string>
#include <iostream>
#include <tuple>
#include <array>
#include <numeric>
#include <algorithm>
#include <cassert>

namespace mio
{

/**
 * Simple date representation as year, month, and day.
 * month in [1, 12], day in [1, 31]
 */
struct Date {
    /**
     * default constructor.
     */
    Date() = default;

    /**
     * initializing constructor.
     * @param y year as int
     * @param m month as int, 1 - 12
     * @param d day as int, 1 - 31
     */
    Date(int y, int m, int d)

        : year(y)
        , month(m)
        , day(d)
    {
        assert(month > 0 && month < 13);
        assert(day > 0 && day < 32);
    }

    /**
     * equality comparison operators.
     * @param other another date.
     */
    //@{
    bool operator==(const Date& other) const
    {
        return year == other.year && month == other.month && day == other.day;
    }
    bool operator!=(const Date& other) const
    {
        return !(*this == other);
    }
    bool operator<(const Date& other) const
    {
        if (std::tie(year, month, day) < std::tie(other.year, other.month, other.day)) {
            return true;
        }
        else {
            return false;
        }
    }
    bool operator<=(const Date& other) const
    {
        return !(other < *this);
    }
    bool operator>(const Date& other) const
    {
        return other < *this;
    }
    bool operator>=(const Date& other) const
    {
        return !(*this < other);
    }
    //@}

    /**
     * gtest printer.
     */
    friend void PrintTo(const Date& self, std::ostream* os)
    {
        *os << self.year << "." << self.month << "." << self.day;
    }

    /**
     * serialize this. 
     * @see mio::serialize
     */
    template <class IOContext>
    void serialize(IOContext& io) const
    {
        auto obj = io.create_object("Date");
        obj.add_element("Year", year);
        obj.add_element("Month", month);
        obj.add_element("Day", day);
    }

    /**
     * deserialize an object of this class.
     * @see mio::deserialize
     */
    template<class IOContext>
    static IOResult<Date> deserialize(IOContext& io)
    {
        auto obj = io.expect_object("Date");
        auto y   = obj.expect_element("Year", Tag<int>{});
        auto m   = obj.expect_element("Month", Tag<int>{});
        auto d   = obj.expect_element("Day", Tag<int>{});
        return apply(
            io,
            [](auto&& y_, auto&& m_, auto&& d_) -> IOResult<Date> {
                if (m_ <= 0 || m_ > 12)
                    return failure(StatusCode::OutOfRange, "Month must be between 1 and 12 (inclusive).");
                if (d_ <= 0 || d_ > 31)
                    return failure(StatusCode::OutOfRange, "Day must be between 1 and 31 (inclusive).");
                return success(Date{y_, m_, d_});
            },
            y, m, d);
    }

    int year;
    int month;
    int day;
};

/**
 * parses a date from a string.
 * uses fixed format YYYY.MM.DD.
 * @param date_str date as a string.
 * @return parsed date.
 */
inline Date parse_date(const std::string& date_str)
{
    Date date;
    date.year  = std::stoi(date_str.substr(0, 4));
    date.month = std::stoi(date_str.substr(5, 2));
    date.day   = std::stoi(date_str.substr(8, 2));
    return date;
}

/**
 * @brief Computes the new date corresponding to a given date and a offset in days.
 * @param date date.
 * @param offset_days offset in days.
 * @return new date that is date + offset_days.
 */
inline Date offset_date_by_days(Date date, int offset_days)
{
    auto year  = date.year;
    auto month = date.month;
    auto day   = date.day;
    assert(month > 0 && month < 13 && day > 0 && day < 32);

    std::array<int, 12> month_len;
    if (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0)) {
        // leap year
        month_len = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    }
    else {
        month_len = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    }
    if (day + offset_days > 0 && day + offset_days <= month_len[month - 1]) {
        return {year, month, day + offset_days};
    }
    else {
        std::array<int, 12> part_sum;
        std::partial_sum(month_len.begin(), month_len.end(), part_sum.begin());

        int day_in_year = day + offset_days;
        if (month > 1) {
            // take month-2 since end of last month has to be found and due to start at 0 of C++ (against January=1 in date format)
            day_in_year += part_sum[month - 2];
        }

        if (day_in_year > 0 && day_in_year <= part_sum[11]) {
            auto iter = std::find_if(part_sum.begin(), part_sum.end(), [day_in_year](auto s) {
                return day_in_year <= s;
            });
            int i     = static_cast<int>(iter - part_sum.begin());
            return {year, i + 1, day_in_year - (i > 0 ? part_sum[static_cast<size_t>(i - 1)] : 0)};
        }
        else {
            if (day_in_year > 0) {
                return offset_date_by_days({year + 1, 1, 1}, day_in_year - part_sum[11] - 1);
            }
            else {
                return offset_date_by_days({year - 1, 12, 31}, day_in_year);
            }
        }
    }
}

/**
 * @brief Computes the day in year based on a given date.
 * @param date date
 * @return day in year, starting January, 1st, with 1.
 */
inline int get_day_in_year(Date date)
{
    auto year  = date.year;
    auto month = date.month;
    auto day   = date.day;
    assert(month > 0 && month < 13 && day > 0 && day < 32);

    if (month > 1) {
        std::array<int, 12> month_len;
        if (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0)) {
            // leap year
            month_len = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        }
        else {
            month_len = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        }

        std::array<int, 12> part_sum;
        std::partial_sum(month_len.begin(), month_len.end(), part_sum.begin());

        // take month-2 since end of last month has to be found and due to start at 0 of C++ (against January=1 in date format)
        int day_in_year = part_sum[month - 2] + day;

        return day_in_year;
    }
    else {
        return day;
    }
}

/**
 * @brief Computes the offset in days given two dates: first date minus second date.
 * @param date1 first date.
 * @param date2 second date.
 * @return offset in days between the two dates.
 */
inline int get_offset_in_days(Date date1, Date date2)
{
    auto year1  = date1.year;
    auto month1 = date1.month;
    auto day1   = date1.day;

    auto year2  = date2.year;
    auto month2 = date2.month;
    auto day2   = date2.day;

    if (year1 == year2 && month1 == month2) {
        return day1 - day2;
    }
    else {
        int day_in_year1 = get_day_in_year(date1);
        int day_in_year2 = get_day_in_year(date2);

        if (year1 < year2) {
            int sum_days = 0;
            for (int i = year1; i < year2; i++) {
                sum_days += 365;
                if (((i % 4 == 0) && (i % 100 != 0)) || (i % 400 == 0)) {
                    sum_days += 1;
                }
            }
            return -(sum_days - day_in_year1) - day_in_year2;
        }
        else if (year1 > year2) {
            int sum_days = 0;
            for (int i = year2; i < year1; i++) {
                sum_days += 365;
                if (((i % 4 == 0) && (i % 100 != 0)) || (i % 400 == 0)) {
                    sum_days += 1;
                }
            }
            return day_in_year1 + sum_days - day_in_year2;
        }
        else {

            return day_in_year1 - day_in_year2;
        }
    }
}

} // end namespace mio

#endif // EPI_UTILS_DATE_H
