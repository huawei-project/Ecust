from .vggnet import VGG

modeldict = {
    'analysis_vgg11_bn': lambda inp, outp, size: VGG(inp, outp, size, 'VGG11', batch_norm=True),
}
