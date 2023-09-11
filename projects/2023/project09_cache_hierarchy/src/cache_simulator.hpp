#ifndef CACHE_SIMULATOR_HPP
#define CACHE_SIMULATOR_HPP

#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cassert>

#include <map>
#include <set>
#include <vector>
#include <stdint.h>
#include <math.h>
#include <bitset>

typedef uint64_t ADDRESS;
typedef uint64_t TAG;
typedef uint64_t SET_INDEX;
typedef uint64_t BLOCK_OFFSET;

// #define VERBOSE
// #define CACHE_TRACKING

typedef std::chrono::time_point<std::chrono::system_clock> TIME;
typedef std::pair<double, double> TWO_LEVEL_HITS;

static const auto RED = "\033[0;31m"; // red print color
static const auto NC = "\033[0m";     // no print color

template <typename T>
inline static void verify(const T &cond, const std::string &msg) {
  if (!cond) {
    std::cerr << RED << msg << NC << std::endl;
    exit(EXIT_FAILURE);
  }
}

static auto PROGRAM_START = std::chrono::system_clock::now();

class CacheLine;
class CacheSet;

typedef bool (*REPLACEMENT_POLICY)(const std::pair<TAG, CacheLine> &,
                                   const std::pair<TAG, CacheLine> &);

class Cache {
  // sets -> S = 2^s where s are the block bits
  const std::size_t S_;
  const std::size_t s_;
  // lines per set (associativity)
  const std::size_t E_;
  // bytes per cache block -> B = 2^b where b are the block bits
  const std::size_t B_;
  const std::size_t b_;
  // address bits
  const std::size_t m_;
  // tag bits
  const std::size_t t_;

  const std::string name_;

  REPLACEMENT_POLICY policy_;

  std::vector<CacheSet *> sets;

  std::size_t accesses_ = 0;
  std::size_t hits_ = 0;
  std::size_t misses_ = 0;
  std::size_t cold_misses_ = 0;

public:
  Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
        const std::string &name, REPLACEMENT_POLICY policy);
  Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
        const std::string &name);
  Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
        REPLACEMENT_POLICY policy);
  Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m);

  ~Cache();

  void print(std::ostream &os) const;

  bool access(const ADDRESS &address);
  bool access(void *ptr);

  void reset_statistics() {

#ifdef VERBOSE
    std::cout << "RESET STATISTICS" << std::endl;
#endif

    accesses_ = 0;
    hits_ = 0;
    misses_ = 0;
    cold_misses_ = 0;
  }

  double get_hit_rate() const {
    double hit_rate;

    if (accesses_ > 0)
      hit_rate = static_cast<double>(hits_) / static_cast<double>(accesses_);
    else
      hit_rate = 0.;

#ifdef VERBOSE
    std::cout << "HIT-RATE: " << hit_rate << std::endl;
#endif

    return hit_rate;
  }

  double get_cold_miss_rate() const;

  void print_statistics() const {
    std::cout << name_ << " ACCESSES: " << accesses_ << " (" << hits_
              << " HITS) -> HIT-RATE: " << get_hit_rate() << std::endl;
    std::cout << name_ << " MISSES: " << misses_ << " (" << cold_misses_
              << " COLD-MISSES) -> COLD-MISS-RATE: " << get_cold_miss_rate()
              << std::endl;
  }

  static void print_address_bits(const ADDRESS &address) {
    std::cout << std::bitset<64>(address) << std::endl;
  }

  static void print_address_bits(void *ptr) {
    Cache::print_address_bits(reinterpret_cast<std::uintptr_t>(ptr));
  }

  std::size_t get_set_size() const { return S_; }

  std::size_t get_line_size() const { return E_; }

  std::size_t get_block_size() const { return B_; }

  std::size_t cache_size_bytes() const { return B_ * E_ * S_; }

  REPLACEMENT_POLICY get_policy() const { return policy_; }

  void flush();

private:
  static bool is_int(double i) { return floor(i) == i; }

  TAG get_tag(const ADDRESS &address) const {
    return (get_tag_mask() & address) >> (s_ + b_);
  }

  ADDRESS get_tag_mask() const {
    return ((static_cast<ADDRESS>(1) << t_) - 1) << (s_ + b_);
  }

  SET_INDEX get_set_index(const ADDRESS &address) const {
    return (get_set_index_mask() & address) >> b_;
  }

  ADDRESS get_set_index_mask() const {
    return ((static_cast<ADDRESS>(1) << s_) - 1) << b_;
  }

  BLOCK_OFFSET get_block_offset(const ADDRESS &address) const {
    return get_block_offset_mask() & address;
  }

  ADDRESS get_block_offset_mask() const {
    return (static_cast<ADDRESS>(1) << b_) - 1;
  }
};

class TwoLevelCache {

  Cache *const l1_;
  Cache *const l2_;

public:
  TwoLevelCache(Cache *const l1, Cache *const l2) : l1_(l1), l2_(l2) {}

  std::pair<bool, bool> access(const ADDRESS &address) {
#ifndef CACHE_TRACKING
    return std::make_pair(false, false);
#endif
    bool l1_hit = l1_->access(address);
    bool l2_hit = l2_->access(address);

    return std::make_pair(l1_hit, l2_hit);
  }

  std::pair<bool, bool> access(void *ptr) {
#ifndef CACHE_TRACKING
    return std::make_pair(false, false);
#endif
    return access(reinterpret_cast<std::uintptr_t>(ptr));
  }

  void reset_statistics() {
#ifdef VERBOSE
    std::cout << "RESET STATISTICS" << std::endl;
#endif

    l1_->reset_statistics();
    l2_->reset_statistics();
  }

  TWO_LEVEL_HITS get_hit_rates() {
    double hit_rate_l1 = l1_->get_hit_rate();
    double hit_rate_l2 = l2_->get_hit_rate();

#ifdef VERBOSE
    std::cout << "L1 HIT-RATE: " << hit_rate_l1
              << ", "
                 "L2 HIT-RATE: "
              << hit_rate_l2 << std::endl;
#endif

    return std::make_pair(hit_rate_l1, hit_rate_l2);
  }

  TWO_LEVEL_HITS get_cold_miss_rates() {
    double cold_miss_rate_l1 = l1_->get_cold_miss_rate();
    double cold_miss_rate_l2 = l2_->get_cold_miss_rate();

#ifdef VERBOSE
    std::cout << "L1 COLD-MISS-RATE: " << cold_miss_rate_l1
              << ", "
                 "L2 COLD-MISS-RATE: "
              << cold_miss_rate_l2 << std::endl;
#endif

    return std::make_pair(cold_miss_rate_l1, cold_miss_rate_l2);
  }

  void print(std::ostream &os) const {
    std::cout
        << "----------------------PRINT TWO-LEVEL-CACHE----------------------"
        << std::endl;
    l1_->print(os);
    l2_->print(os);
    std::cout
        << "--------------------END PRINT TWO-LEVEL-CACHE--------------------"
        << std::endl;
  }
};

class CacheLine {

private:
  bool valid_;
  TIME last_access_;
  std::size_t accesses_;

public:
  CacheLine() : valid_(true), accesses_(1) {
    last_access_ = std::chrono::system_clock::now();
  }

  bool is_vaild() const { return valid_; }

  TIME get_last_access() const { return last_access_; }

  std::size_t get_accesses() const { return accesses_; }

  bool access();

  void reset() {
    valid_ = true;
    accesses_ = 1;
    last_access_ = std::chrono::system_clock::now();
  }
};

class CacheSet {

private:
  std::map<TAG, CacheLine> cache_lines_;
  std::set<TAG> cache_lines_history_;
  Cache *const parent_cache_;

  void add_cache_line(const TAG &tag);
  void replace();

public:
  static bool policy_LRU(const std::pair<TAG, CacheLine> &evict,
                         const std::pair<TAG, CacheLine> &keep) {
    return evict.second.get_last_access() < keep.second.get_last_access();
  }

  static bool policy_LFU(const std::pair<TAG, CacheLine> &evict,
                         const std::pair<TAG, CacheLine> &keep) {
    if (evict.second.get_accesses() < keep.second.get_accesses())
      return true;
    else if (evict.second.get_accesses() == keep.second.get_accesses())
      return evict.second.get_last_access() < keep.second.get_last_access();
    return false;
  }

  void print(std::ostream &os) const {
    std::size_t i = 0;
    for (auto const &line : cache_lines_) {
      os << "\t LINE[" << i << "] = " << line.first << "\t(last access: "
         << std::chrono::duration<double, std::milli>(
                line.second.get_last_access() - PROGRAM_START)
                .count()
         << ",\t accesses: " << line.second.get_accesses() << ")" << std::endl;
      ++i;
    }
  }

  bool contains_a_line() const { return cache_lines_.size() > 0; }

  CacheSet(Cache *const cache) : parent_cache_(cache) {}

  std::pair<bool, bool> access(const TAG &tag);
};

#endif
