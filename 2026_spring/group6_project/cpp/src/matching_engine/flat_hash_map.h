#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace ghostbook::matching {

template <typename Key,
          typename Value,
          typename Hasher = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class FlatHashMap {
public:
	FlatHashMap() = default;

	explicit FlatHashMap(std::size_t capacity) { reserve(capacity); }

	bool contains(const Key& key) const { return find_slot(key) != slots_.size(); }

	Value* find(const Key& key) {
		auto index = find_slot(key);
		if (index == slots_.size()) {
			return nullptr;
		}
		return &slots_[index].entry->second;
	}

	const Value* find(const Key& key) const {
		auto index = find_slot(key);
		if (index == slots_.size()) {
			return nullptr;
		}
		return &slots_[index].entry->second;
	}

	Value& operator[](const Key& key) {
		if (auto* existing = find(key)) {
			return *existing;
		}
		return emplace_value(key, Value{});
	}

	template <typename T>
	Value& insert_or_assign(const Key& key, T&& value) {
		if (needs_rehash(size_ + 1)) {
			rehash(slots_.empty() ? initial_capacity_ : slots_.size() << 1U);
		}

		return insert_or_assign_no_rehash(key, std::forward<T>(value));
	}

	bool erase(const Key& key) {
		auto index = find_slot(key);
		if (index == slots_.size()) {
			return false;
		}
		slots_[index].entry.reset();
		slots_[index].tombstone = true;
		--size_;
		if (tombstones_ < slots_.size()) {
			++tombstones_;
		}
		return true;
	}

	void reserve(std::size_t capacity) {
		if (capacity <= slots_.size()) {
			return;
		}
		rehash(next_capacity(capacity));
	}

	std::size_t size() const { return size_; }

private:
	struct Slot {
		std::optional<std::pair<Key, Value>> entry;
		bool tombstone{false};
	};

	static constexpr float max_load_factor_ = 0.70F;
	static constexpr std::size_t initial_capacity_ = 16;

	std::vector<Slot> slots_;
	std::size_t size_{0};
	std::size_t tombstones_{0};
	Hasher hasher_{};
	KeyEqual key_equal_{};

	std::size_t next_capacity(std::size_t requested) const {
		std::size_t capacity = initial_capacity_;
		while (capacity < requested) {
			capacity <<= 1U;
		}
		return capacity;
	}

	bool needs_rehash(std::size_t prospective_size) const {
		if (slots_.empty()) {
			return true;
		}
		return static_cast<float>(prospective_size + tombstones_) / static_cast<float>(slots_.size()) >
		       max_load_factor_;
	}

	std::size_t probe_for_insert(const Key& key) {
		if (slots_.empty()) {
			rehash(initial_capacity_);
		}

		std::size_t mask = slots_.size() - 1;
		std::size_t index = hasher_(key) & mask;
		std::size_t first_tombstone = slots_.size();
		while (true) {
			auto& slot = slots_[index];
			if (!slot.entry.has_value()) {
				return first_tombstone != slots_.size() ? first_tombstone : index;
			}
			if (key_equal_(slot.entry->first, key)) {
				return index;
			}
			if (slot.tombstone && first_tombstone == slots_.size()) {
				first_tombstone = index;
			}
			index = (index + 1U) & mask;
		}
	}

	std::size_t find_slot(const Key& key) const {
		if (slots_.empty()) {
			return slots_.size();
		}

		std::size_t mask = slots_.size() - 1;
		std::size_t index = hasher_(key) & mask;
		while (true) {
			auto const& slot = slots_[index];
			if (!slot.entry.has_value()) {
				if (!slot.tombstone) {
					return slots_.size();
				}
			} else if (key_equal_(slot.entry->first, key)) {
				return index;
			}
			index = (index + 1U) & mask;
		}
	}

	template <typename T>
	Value& emplace_value(const Key& key, T&& value) {
		if (needs_rehash(size_ + 1)) {
			rehash(slots_.empty() ? initial_capacity_ : slots_.size() << 1U);
		}
		return insert_or_assign_no_rehash(key, std::forward<T>(value));
	}

	template <typename T>
	Value& insert_or_assign_no_rehash(const Key& key, T&& value) {
		auto index = probe_for_insert(key);
		auto& slot = slots_[index];
		if (slot.entry.has_value()) {
			slot.entry->second = std::forward<T>(value);
			return slot.entry->second;
		}

		slot.entry.emplace(key, std::forward<T>(value));
		slot.tombstone = false;
		++size_;
		return slot.entry->second;
	}

	void rehash(std::size_t new_capacity) {
		new_capacity = next_capacity(new_capacity);
		std::vector<Slot> old_slots = std::move(slots_);
		slots_.clear();
		slots_.resize(new_capacity);
		size_ = 0;
		tombstones_ = 0;

		for (auto& slot : old_slots) {
			if (!slot.entry.has_value()) {
				continue;
			}
			insert_or_assign_no_rehash(slot.entry->first, std::move(slot.entry->second));
		}
	}
};

}  // namespace ghostbook::matching