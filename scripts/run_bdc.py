import argparse
import importlib

PROFILE_MAP = {
    'MFIC': 'bdc_profiles.mfic',
    'ARCC': 'bdc_profiles.arcc',
    'OBDC': 'bdc_profiles.obdc',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    args, passthrough = p.parse_known_args()

    ticker = args.ticker.upper()
    module_name = PROFILE_MAP.get(ticker, 'bdc_analyzer')
    mod = importlib.import_module(module_name)

    if not hasattr(mod, 'analyze'):
        raise RuntimeError(f'Module {module_name} has no analyze(ticker)')

    mod.analyze(ticker)


if __name__ == '__main__':
    main()
