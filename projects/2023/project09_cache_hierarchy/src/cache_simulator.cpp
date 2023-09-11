#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdint>

#include "cache_simulator.hpp"

Cache::Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
             const std::string &name, REPLACEMENT_POLICY policy)
    : S_(S), s_(static_cast<std::size_t>(log2(S))), E_(E), B_(B),
      b_(static_cast<std::size_t>(log2(B))), m_(m), t_(m_ - s_ - b_),
      name_(name), policy_(policy) {

  verify(is_int(log2(S)), "S is not a power of two");
  verify(is_int(log2(B)), "B is not a power of two");
  verify(m > 0, "Address consists of zero bits");

  for (std::size_t i = 0; i < S_; ++i) {
    sets.push_back(new CacheSet(this));
  }

#ifdef VERBOSE
  std::cout << "CACHE CONSTRUCTOR: SIZE " << cache_size_bytes()
            << " BYTES -> ADRESS-TRANSLATION = [" << t_ << "," << s_ << ","
            << b_ << "]" << std::endl;
#endif
}

Cache::Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
             const std::string &name)
    : Cache(S, E, B, m, name, &CacheSet::policy_LRU) {}
Cache::Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m,
             REPLACEMENT_POLICY policy)
    : Cache(S, E, B, m, "", policy) {}
Cache::Cache(std::size_t S, std::size_t E, std::size_t B, std::size_t m)
    : Cache(S, E, B, m, "", &CacheSet::policy_LRU) {}

Cache::~Cache() {
  for (std::size_t i = 0; i < S_; ++i) {
    delete sets[i];
  }
}

bool Cache::access(const ADDRESS &address) {
#ifndef CACHE_TRACKING
  return false;
#endif
  TAG tag = get_tag(address);
  SET_INDEX set_index = get_set_index(address);

#ifdef VERBOSE
  std::cout << name_;
#endif

  bool ret;
  std::pair<bool, bool> acc = sets[set_index]->access(tag);
  bool is_hit = acc.first;
  bool is_cold_miss = acc.second;

  if (is_hit) {
#ifdef VERBOSE
    std::cout << " ACCESS: HIT  ->\t";
#endif
    ++hits_;
    ret = true;
  } else {
#ifdef VERBOSE
    if (is_cold_miss) {
      std::cout << " ACCESS: COLD MISS ->\t";
    } else {
      std::cout << " ACCESS: MISS ->\t";
    }
#endif

    if (is_cold_miss)
      ++cold_misses_;
    ++misses_;
    ret = false;
  }

#ifdef VERBOSE
  BLOCK_OFFSET block_offset = get_block_offset(address);
  std::cout << "[" << set_index << "," << block_offset << "] = " << tag
            << "\t (ADDRESS AS INTEGER = " << address << ")" << std::endl;
#endif

  ++accesses_;
  return ret;
}

bool Cache::access(void *ptr) {
#ifndef CACHE_TRACKING
  return false;
#endif
  return access(reinterpret_cast<std::uintptr_t>(ptr));
}

void Cache::print(std::ostream &os) const {
  std::cout << "------------------PRINT " << name_ << " CACHE------------------"
            << std::endl;
  for (std::size_t i = 0; i < sets.size(); ++i) {
    if (sets[i]->contains_a_line()) {
      std::cout << "SET[" << i << "]" << std::endl;
      sets[i]->print(os);
    }
  }
  std::cout << "----------------END PRINT " << name_ << " CACHE----------------"
            << std::endl;
}

double Cache::get_cold_miss_rate() const {
  double cold_miss_rate;

  if (misses_ > 0)
    cold_miss_rate =
        static_cast<double>(cold_misses_) / static_cast<double>(misses_);
  else
    cold_miss_rate = 0.;

#ifdef VERBOSE
  std::cout << "COLD-MISS-RATE: " << cold_miss_rate << std::endl;
#endif

  return cold_miss_rate;
}

void Cache::flush() {
  for (std::size_t i = 0; i < sets.size(); ++i) {
    delete sets[i];
    sets[i] = new CacheSet(this);
  }
  reset_statistics();
}

bool CacheLine::access() {
  if (!is_vaild())
    return false;
  last_access_ = std::chrono::system_clock::now();
  ++accesses_;
  return true;
}

void CacheSet::replace() {
  std::vector<std::pair<TAG, CacheLine>> cache_vec(cache_lines_.begin(),
                                                   cache_lines_.end());
  std::sort(cache_vec.begin(), cache_vec.end(), parent_cache_->get_policy());
  cache_lines_.erase(cache_vec[0].first);
}

std::pair<bool, bool> CacheSet::access(const TAG &tag) {
  bool is_cold_miss = false;
  if (!cache_lines_history_.contains(tag)) {
    cache_lines_history_.insert(tag);
    is_cold_miss = true;
  }

  bool is_hit;
  auto search = cache_lines_.find(tag);
  if (search != cache_lines_.end() && search->second.is_vaild()) {
    search->second.access();
    is_hit = true;
  } else {

    if (search == cache_lines_.end()) {
      add_cache_line(tag);
    } else { // was not valid
      search->second.reset();
    }
    is_hit = false;
  }

  assert(!is_cold_miss || !is_hit); // cold miss -> no hit

  return std::make_pair(is_hit, is_cold_miss);
}

void CacheSet::add_cache_line(const TAG &tag) {
  if (cache_lines_.size() == parent_cache_->get_line_size()) {
    replace();
  }
  cache_lines_.insert({tag, CacheLine()});
}
