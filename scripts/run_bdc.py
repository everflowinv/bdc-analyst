import argparse
import importlib
import inspect

PROFILE_MAP = {
    'MFIC': 'bdc_profiles.mfic',
    'ARCC': 'bdc_profiles.arcc',
    'OBDC': 'bdc_profiles.obdc',
    'BXSL': 'bdc_profiles.bxsl',
    'MAIN': 'bdc_profiles.main',
    'GBDC': 'bdc_profiles.gbdc',
    'FSK': 'bdc_profiles.fsk',
    'HTGC': 'bdc_profiles.htgc',
    'TSLX': 'bdc_profiles.tslx',
    'CSWC': 'bdc_profiles.cswc',
    'PSEC': 'bdc_profiles.psec',
    'MSDL': 'bdc_profiles.msdl',
    'OTF': 'bdc_profiles.otf',
    'TRIN': 'bdc_profiles.trin',
    'GSBD': 'bdc_profiles.gsbd',
    'OCSL': 'bdc_profiles.ocsl',
    'KBDC': 'bdc_profiles.kbdc',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    p.add_argument('--periodA', required=False)
    p.add_argument('--periodB', required=False)
    args, passthrough = p.parse_known_args()

    ticker = args.ticker.upper()
    module_name = PROFILE_MAP.get(ticker, 'bdc_analyzer')
    mod = importlib.import_module(module_name)

    if not hasattr(mod, 'analyze'):
        raise RuntimeError(f'Module {module_name} has no analyze(ticker)')

    sig = inspect.signature(mod.analyze)
    if 'periodA' in sig.parameters and 'periodB' in sig.parameters:
        mod.analyze(ticker, periodA=args.periodA, periodB=args.periodB)
    else:
        mod.analyze(ticker)


if __name__ == '__main__':
    main()
