"""
copied modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py
"""
import os
from pathlib import Path

import imageio
import numpy as np
import pickle

import dominate
from dominate.tags import meta, h1, h3, table, tr, td, p, a, img, br
import torchvision
import torch
from torchvision.io import write_video


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """
    def __init__(self,
                 web_dir,
                 title,
                 refresh=0,
                 cache=False,
                 resume=False,
                 reverse=False):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.refresh = refresh
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.use_cache = cache
        self.cache_file = os.path.join(self.web_dir, 'cache.pkl')
        self.cache = []
        self.reverse = reverse

        if resume and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            self.rebuild_()
        else:
            self.doc = dominate.document(title=title)
            if refresh > 0:
                with self.doc.head:
                    meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def rebuild_(self):
        if getattr(self, 'doc', None):
            del self.doc
        self.doc = self._create_doc(self.title, self.refresh)
        self.doc = self._build_from_cache(self.doc, self.cache, self.reverse)
        return self.doc

    def _create_doc(self, title, refresh=0):
        doc = dominate.document(title=title)
        if refresh > 0:
            with doc.head:
                meta(http_equiv="refresh", content=str(refresh))
        with doc:
            h1(title)
        return doc

    def _build_from_cache(self, doc, cache, reverse=False):
        if reverse:
            cache = cache[::-1]
        for item in cache:
            if item[0] == 'header':
                self._add_header(doc, *item[1:])
            elif item[0] == 'images':
                self._add_images(doc, *item[1:])
            else:
                raise NotImplementedError
        return doc

    def _add_header(self, doc, text):
        with doc:
            h3(text)

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        if self.use_cache:
            self.cache.append(('header', text))
        else:
            self._add_header(self.doc, text)

    def _add_images(self, doc, ims, txts, links, width=400):
        t = table(border=1, style="table-layout: fixed;")  # Insert a table
        doc.add(t)
        with t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;",
                            halign="center",
                            valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width,
                                    src=os.path.join('images', im))
                            br()
                            p(txt)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        if self.use_cache:
            self.cache.append(('images', ims, txts, links, width))
        else:
            self._add_images(self.doc, ims, txts, links, width)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = os.path.join(self.web_dir, 'index.html')
        if self.use_cache:
            self.rebuild_()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        with open(html_file, 'wt') as f:
            f.write(self.doc.render())


def initialize_webpage(web_dir, name='dalle', resume=False, reverse=True):
    webpage = HTML(web_dir,
                   name,
                   resume=resume,
                   cache=True,
                   reverse=reverse,
                   refresh=0)
    return webpage


@torch.no_grad()
def save_image_tensor(tensor, path, video_format='gif'):
    tensor = tensor.squeeze(0)
    if len(tensor.shape) == 3 or (len(tensor.shape) == 4
                                  and tensor.shape[0] == 1):
        # Save image
        path = str(path) + '.png'
        torchvision.utils.save_image(tensor,
                                     path,
                                     normalize=True,
                                     range=(0, 1))
    elif len(tensor.shape) == 4 or (len(tensor.shape) == 5
                                    and tensor.shape[0] == 1):
        if video_format == 'gif':
            # Save gif
            path = str(path) + '.gif'
            imageio.mimsave(
                path,
                (tensor.data.cpu().clamp(0, 1) * 255).type(
                    torch.uint8).permute(0, 2, 3, 1).numpy(),
            )
        else:
            # Save mp4
            path = str(path) + '.mp4'
            write_video(
                path,
                (tensor.data.cpu().clamp(0, 1) * 255).type(
                    torch.uint8).permute(0, 2, 3, 1),
                fps=4,
            )
    else:
        raise RuntimeError
    name = Path(path).name
    return name


@torch.no_grad()
def save_grid(webpage=None,
              tensor=None,
              caption=None,
              name='',
              nrow=1,
              width=256,
              video_format='gif'):
    """assuming each row contains multiple samples of one text, if nrow == 1 save all samples in a row"""
    img_dir = Path(webpage.get_image_dir())
    if isinstance(nrow, list):
        n_per_row = nrow
    else:
        n_per_row = [nrow] * (len(tensor) // nrow)
    n_row = len(n_per_row)

    imgs = []
    cumsum = np.cumsum([0] + n_per_row)
    for i in range(n_row):
        for j in range(n_per_row[i]):
            idx = cumsum[i] + j
            img_name = save_image_tensor(
                tensor[idx].detach().cpu(),
                img_dir / (name + f'_{i+1}_{j+1}'),
                video_format,
            )
            imgs.append(img_name)

    if nrow == 1:
        webpage.add_images(imgs, caption, imgs, width=width)
    else:
        for i in range(n_row):
            imgs_row = imgs[cumsum[i]:cumsum[i + 1]]
            txts_row = caption[cumsum[i]:cumsum[i + 1]]
            webpage.add_images(imgs_row, txts_row, imgs_row, width=width)

    webpage.save()


if __name__ == '__main__':
    # An example usage
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
