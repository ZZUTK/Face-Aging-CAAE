import sys, os

kilobytes = 1024
megabytes = kilobytes * 1000
chunksize = int(24 * megabytes)                   # default: the upload limit of Github


def split(fromfile, todir, chunksize=chunksize): 
    if not os.path.exists(todir):                  # caller handles errors
        os.mkdir(todir)                            # make dir, read/write parts
    else:
        for fname in os.listdir(todir):            # delete any existing files
            os.remove(os.path.join(todir, fname)) 
    partnum = 0
    input = open(fromfile, 'rb')                   # use binary mode on Windows
    while 1:                                       # eof=empty string from read
        chunk = input.read(chunksize)              # get next part <= chunksize
        if not chunk: break
        partnum  = partnum+1
        filename = os.path.join(todir, ('part%04d' % partnum))
        fileobj  = open(filename, 'wb')
        fileobj.write(chunk)
        fileobj.close()                            # or simply open(  ).write(  )
    input.close()
    assert partnum <= 9999                         # join sort fails if 5 digits
    return partnum


def join(fromdir, tofile):
    readsize = 1024
    output = open(tofile, 'wb')
    parts  = os.listdir(fromdir)
    parts.sort()
    for filename in parts:
        filepath = os.path.join(fromdir, filename)
        fileobj  = open(filepath, 'rb')
        while 1:
            filebytes = fileobj.read(readsize)
            if not filebytes: break
            output.write(filebytes)
        fileobj.close()
    output.close()


if __name__ == '__main__':
    # split a file to multiple parts
    # try:
    #     parts = split('init_model/model-init.data-00000-of-00001', 'init_model/model_parts')
    # except:
    #     print 'Error during split:'
        
    # join multiple parts to the original file
    try:
        join('init_model/model_parts', 'init_model/model-init.data-00000-of-00001')
    except:
        print 'Error joining files:'
        
       
