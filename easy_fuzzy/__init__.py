def classFactory(iface):
    from .plugin import EasyFuzzyPlugin
    return EasyFuzzyPlugin(iface)