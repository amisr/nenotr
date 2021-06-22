# -*- coding: utf-8 -*-
# Created: 4 July 2018
# Author: Ashton S. Reimer

import os
import tables

# Class for repacking h5 files and compressing them
ignore_attribs = ['TITLE','CLASS','VERSION']
class Repackh5(object):
    """
    Repack the input hdf5 file and compress the arrays
    """
    def __init__(self,input_fname,output_fname):
        self.input_file = input_fname
        self.output_file = output_fname

    def repack(self):
        FILTERS = tables.Filters(complib='zlib', complevel=1)
        with tables.open_file(self.input_file,'r') as input_h5:
            with tables.open_file(self.output_file,'w',filters=FILTERS) as output_h5:

                for group in input_h5.walk_groups():
                    group_name = group._v_pathname

                    # First make sure the group in the input file is in the output file.
                    output_h5_groups = [g._v_pathname for g in output_h5.walk_groups()]
                    if group_name not in output_h5_groups:
                        root, name = os.path.split(group_name)
                        output_h5.create_group(root,name)

                    # Now list the nodes in the group. For any node that isn't a group, write it
                    # to the output file.
                    for node in input_h5.list_nodes(group_name):
                        if node._v_attrs.CLASS != 'GROUP':
                            # Read the array, get it's attributes
                            name = node.name
                            title = node.title
                            data = node.read()
                            try:
                                shape = data.shape
                                is_array = True
                            except AttributeError:
                                shape = ()
                                is_array = False
                            
                            #attributes
                            output_attribs = dict()
                            input_attribs = node.attrs.__dict__
                            output_attribs_keys = [x for x in input_attribs.keys() if not x.startswith('_') and not x in ignore_attribs]
                            for key in output_attribs_keys:
                                output_attribs[key] = input_attribs[key]


                            # Write the array, write it's attributes
                            if len(shape) and is_array:
                                atom = tables.Atom.from_dtype(data.dtype)
                                array = output_h5.create_carray(group_name,name,atom,shape,title=title)
                                array[:] = data
                            else:
                                output_h5.create_array(group_name,name,data,title=title)

                            for key in output_attribs_keys:
                                output_h5.set_node_attr(node._v_pathname,key,output_attribs[key])

