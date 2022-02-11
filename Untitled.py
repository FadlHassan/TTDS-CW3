from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP
from ipaddress import ip_address, ip_network
import sys
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, ip, plen):
        self.bytes = plen
        self.left = None
        self.right = None
        self.ip = ip

    def add(self, ip, plen):

        if ip == self.ip:
            aggregate = plen + self.bytes
            self.bytes = aggregate # if IP equals then aggregate

        elif self.ip > ip:

            if not (self.left):   # if didn't have left node then create.
                obj = Node(ip, plen)
                self.left = obj


            else: # if it has left node then add

                self.left.add(ip, plen)

        else:

            if not (self.right):  # if didn't have right node then create
                obj = Node(ip, plen)
                self.right = obj

            else: # if it has right node then add
                self.right.add(ip, plen)


    def data(self, data):
        if self.left:
            self.left.data(data)
        if self.bytes > 0:
            data[ip_network(self.ip)] = self.bytes
        if self.right:
            self.right.data(data)

    @staticmethod
    def supernet(ip1, ip2):
        # arguments are either IPv4Address or IPv4Network
        na1 = ip_network(ip1).network_address
        na2 = ip_network(ip2).network_address

        binary1 = bin(int(na1))[2:]
        binary2 = bin(int(na2))[2:]

        binary1 = binary1.zfill(32) # convert to 32 bits
        binary2 = binary2.zfill(32) # convert to 32 bits

        net_mask = 0
        common_bin = '0b'

        while (binary1[net_mask] == binary2[net_mask]) and net_mask != 32:

            common_bin = common_bin + binary1[net_mask]
            net_mask = net_mask + 1 # for calculating number of prefix bits (EQUAL)

        if '0b' == common_bin:
            common_addr = 0 # if no common found at the first bit we put common address equal to zero

        else:

            # enlarging the common address
            masking = 2 ** (32 - net_mask)
            temp = int(common_bin, 2)
            common_addr = temp * masking


        string_ip = str(ip_address(common_addr))

        return ip_network('{}/{}'.format(string_ip, net_mask), strict=False)

    def aggr(self, byte_thresh):

        if self.left != None:

            self.left.aggr(byte_thresh) # Recursion

            if byte_thresh > self.left.bytes:

                self.bytes += self.left.bytes # Aggregate bytes
                self.ip = Node.supernet(self.ip, self.left.ip)  # Supernet IP
                self.left.bytes = 0

                bool1 = (self.left.right == None)
                bool2 = (self.left.left == None)

                if bool1 and bool2:

                    self.left = None # satisfied (true) condition implies leaf node henceforth delete it

        if self.right != None:

            self.right.aggr(byte_thresh) # recursion post-order

            if byte_thresh > self.right.bytes:

                self.bytes += self.right.bytes # Aggregate bytes
                self.ip = Node.supernet(self.ip, self.right.ip) # Supernet IP
                self.right.bytes = 0

                bool1 = (self.right.left == None)
                bool2 = (self.right.right == None)

                if bool1 and bool2:

                    self.right  = None # satisfied (true) condition implies leaf node henceforth delete it


class Data(object):
    def __init__(self, data):
        self.tot_bytes = 0
        self.data = {}
        self.aggr_ratio = 0.05
        root = None
        cnt = 0
        for pkt, metadata in RawPcapReader(data):
            ether = Ether(pkt)
            if not 'type' in ether.fields:
                continue
            if ether.type != 0x0800:
                continue
            ip = ether[IP]
            self.tot_bytes += ip.len
            if root is None:
                root = Node(ip_address(ip.src), ip.len)
            else:
                root.add(ip_address(ip.src), ip.len)
            cnt += 1
        root.aggr(self.tot_bytes * self.aggr_ratio)
        root.data(self.data)
    def Plot(self):
        data = {k: v/1000 for k, v in self.data.items()}
        plt.rcParams['font.size'] = 8
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(which='major', axis='y')
        ax.tick_params(axis='both', which='major')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(l) for l in data.keys()], rotation=45,
            rotation_mode='default', horizontalalignment='right')
        ax.set_ylabel('Total bytes [KB]')
        ax.bar(ax.get_xticks(), data.values(), zorder=2)
        ax.set_title('IPv4 sources sending {} % ({}KB) or more traffic.'.format(
            self.aggr_ratio * 100, self.tot_bytes * self.aggr_ratio / 1000))
        plt.savefig(sys.argv[1] + '.aggr.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    d = Data(sys.argv[1])
    d.Plot()
