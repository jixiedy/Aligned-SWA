import heapq
import random

class BtmkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []
 
    def Push(self, elem):
        # Reverse elem to convert to max-heap
        elem = -elem
        # Using heap algorighem
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)
 
    def BtmK(self):
        return sorted([-x for x in self.data])
    

if __name__ == "__main__":
    print("Hello")
    list_rand = random.sample(xrange(1000000), 100)
    th = BtmkHeap(3)
    for i in list_rand:
        th.Push(i)
    print(th.BtmK())
    print(sorted(list_rand)[0:3])
    