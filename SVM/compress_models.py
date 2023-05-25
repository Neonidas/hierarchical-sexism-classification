import pickle
import sys
import bz2
import oz

filename = sys.argv[1]
print("Loading model...")
loaded_model = pickle.load(open(filename, 'rb'))

print("Trying to compress file")
ofile = bz2.BZ2File(filename+"_compressed_bz2", 'wb')
pickle.dump(loaded_model,ofile)
ofile.close()

