#!/usr/bin/python
# coding: utf-8

# In[ ]:


import cgi, os
import cgitb; cgitb.enable()
import daun
form = cgi.FieldStorage()
# Get filename here.
fileitem = form['filename']
# Test if the file was uploaded
if fileitem.filename:
   # strip leading path from file name to avoid
   # directory traversal attacks
   fn = os.path.basename(fileitem.filename.replace("\\", "/" ))
   open('/tmp/' + fn, 'wb').write(fileitem.file.read())
   message = 'The file "' + fn + '" was uploaded successfully'
else:
   message = 'No file was uploaded'
print """Content-Type: text/html\n
<html>
<body>
   <p>daun.str(np.max(result))+"\n"+daun.label_binarizer.classes_[itemindex[1][0]]</p>
</body>
</html>
""" % (message,)

