import heapq

class QueueItem:
    def __init__(self, bound, list_of_indexes, status="waiting"):
        self.bound = bound
        self.list_of_indexes = list_of_indexes
        self.status = status  # "waiting", "finished"

    def __lt__(self, other):
        return self.bound > other.bound  # max-heap behavior

    def __repr__(self):
        return f"(bound={self.bound}, status={self.status}, indexes={self.list_of_indexes})"


class TopQueue:
    def __init__(self):
        self.heap = []
        self.index_map = {}  # maps index tuple → QueueItem

    def push(self, bound, list_of_indexes, status="waiting"):
        item = QueueItem(bound, list_of_indexes, status)
        heapq.heappush(self.heap, item)
        self.index_map[tuple(list_of_indexes)] = item

    def pop(self):
        while self.heap:
            item = heapq.heappop(self.heap)
            key = tuple(item.list_of_indexes)
            if key in self.index_map:
                del self.index_map[key]
                return item
        return None

    def update_status(self, list_of_indexes, new_status):
        key = tuple(list_of_indexes)
        if key in self.index_map:
            self.index_map[key].status = new_status

    def get_status(self, list_of_indexes):
        key = tuple(list_of_indexes)
        return self.index_map[key].status if key in self.index_map else None

    def is_empty(self):
        return len(self.heap) == 0
