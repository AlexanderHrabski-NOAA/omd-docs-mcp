"""Import Dask, dask-jobqueue, ecFlow, UFS, CICE, FV3 and MPAS documentation
snapshots, and crawl the authoritative MOM6 site, preserving other collections.

Each project is a separate library. Documentation is imported from an immutable
upstream ref when the repository holds the authoritative source; MOM6 is crawled
from its official rendered site because its Doxygen API reference exists only
there. License/attribution files are kept beside each corpus, and rolling branches
are labelled as snapshots, never as releases.
"""
import argparse
import base64
import datetime
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from fetch_pytorch import crawl_site, download_html_zip

ROOT = Path(__file__).resolve().parent.parent

# MOM6 is acquired from its official /en/main/ HTML site (build-time, like PyTorch),
# because the repository docs/ tree only holds Doxygen stubs for the API reference.
MOM6_SITES = {
    'mom6': {
        'version': 'snapshot-main',
        'base_url': 'https://mom6.readthedocs.io/en/main/',
        'start_pages': ('index.html',),
        'path': 'corpus/mom6/source',
        'htmlzip_url': 'https://mom6.readthedocs.io/_/downloads/en/main/htmlzip/',
        'revision': '0e9a5611a9f6c99bfea12665da3a9146183521ee',
        'url': 'https://github.com/mom-ocean/MOM6',
        'version_basis': ('Official rolling /en/main/ documentation site (authoritative for MOM6), '
                          'acquired from the rendered site (htmlzip artifact when published, polite '
                          'crawl otherwise); includes the Doxygen API reference '
                          'generated only on the rendered site; main branch observed at commit '
                          '0e9a5611a9f6c99bfea12665da3a9146183521ee when pinned; not release-certified'),
    },
}

# Git-snapshot libraries: authoritative documentation source lives in the repository.
SNAPSHOT = {
    'dask': {
        'repo': 'dask/dask', 'ref': '2026.8.0', 'commit': '9dc535daa30d7d36d63d4a003b54342fbc7032db',
        'version': '2026.8.0', 'doc_dir': 'docs/source', 'exts': ('.rst', '.md', '.txt'),
        'licenses': ['LICENSE.txt'],
        'version_basis': ('Official release tag 2026.8.0; documentation source imported from the tagged '
                          'repository tree; autosummary API pages are stubs in source and expand only on '
                          'the rendered site'),
    },
    'dask-jobqueue': {
        'repo': 'dask/dask-jobqueue', 'ref': '0.9.0', 'commit': 'b91983d8192c5baccbd7dd0a358f66e0987a97ab',
        'version': '0.9.0', 'doc_dir': 'docs/source', 'exts': ('.rst', '.md', '.txt'),
        'licenses': ['LICENSE.txt'],
        'version_basis': 'Official release tag 0.9.0; documentation source from the tagged repository tree',
    },
    'ecflow': {
        'repo': 'ECMWF/ecFlow', 'ref': '5.19.0', 'commit': 'b6c3ef73323eb0a6d82586fed080641810b11716',
        'version': '5.19.0', 'doc_dir': 'docs', 'exts': ('.rst', '.md', '.txt'),
        'licenses': ['LICENSE', 'NOTICE'],
        'version_basis': ('Official release tag 5.19.0; documentation source from the tagged repository '
                          'tree, including the committed Python and REST client API pages'),
    },
    'ufs-weather-model': {
        'repo': 'ufs-community/ufs-weather-model', 'ref': 'Release/P8b',
        'commit': '9b6b7402de2c06eaf121c4a9a4285b651a936c5d',
        'version': 'P8b', 'doc_dir': 'doc/UsersGuide/source', 'exts': ('.rst', '.md', '.txt', '.inc'),
        'licenses': ['LICENSE.md'],
        'version_basis': ('Official release tag Release/P8b; Users Guide source from the tagged repository '
                          'tree; the guide itself declares no version number'),
    },
    'cice': {
        'repo': 'CICE-Consortium/CICE', 'ref': 'CICE6.6.3', 'commit': 'c734c12bfb19632cd700cb06b22c3ea62b4adeae',
        'version': '6.6.3', 'doc_dir': 'doc/source', 'exts': ('.rst', '.md', '.txt'),
        'licenses': ['LICENSE.pdf', 'COPYRIGHT.pdf'],
        'version_basis': ('Official release tag CICE6.6.3; documentation source from the tagged repository '
                          'tree; BSD-3-Clause license retained as LICENSE.pdf'),
    },
    'fv3': {
        'repo': 'NOAA-GFDL/GFDL_atmos_cubed_sphere', 'ref': 'FV3-202604-public',
        'commit': '6f658bd00ffa52158c1c0f32244b6dd59a0360d7',
        'version': '202604', 'doc_dir': 'docs/doc_source', 'exts': ('.tex', '.md', '.txt'),
        'licenses': ['LICENSE.md'],
        'version_basis': ('Official release tag FV3-202604-public (202604 release); LaTeX technical guide '
                          'source from the tagged repository tree; Jupyter notebooks in docs/examples are '
                          'not indexed'),
        'examples': {'doc_dir': 'docs/examples', 'exts': ('.F90', '.h', '.txt', '.md'),
                     'version_basis': ('Fortran helpers and README files from docs/examples at the tagged '
                                        'repository tree; Jupyter notebooks are not copied because they are '
                                        'not indexable by this server')},
    },
    'mpas': {
        'repo': 'MPAS-Dev/MPAS-Documents', 'ref': 'master', 'commit': '69b46ed167481ca5732636be219c608f96b89121',
        'version': 'snapshot-69b46ed16748', 'target_release': '8.4.2', 'doc_dir': '.',
        'exts': ('.tex', '.md', '.txt'), 'licenses': [],
        'version_basis': ('Rolling master snapshot of MPAS-Dev/MPAS-Documents, the authoritative MPAS '
                          'documentation repository; not release-certified; corresponds to MPAS-Model 8.4.2; '
                          'the repository has no license file'),
    },
}

# Per-library tag -> version transform (strips release prefixes the tags carry).
TAG_VERSION = {
    'dask': lambda tag: tag,
    'dask-jobqueue': lambda tag: tag,
    'ecflow': lambda tag: tag,
    'ufs-weather-model': lambda tag: tag.split('/')[-1],
    'cice': lambda tag: re.sub(r'^CICE', '', tag),
    'fv3': lambda tag: re.sub(r'^FV3-|-public$', '', tag),
}


def tag_version(library, tag):
    transform = TAG_VERSION.get(library)
    return transform(tag) if transform else tag


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def api_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'omd-docs-mcp/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def latest_ref(library):
    """Resolve the newest ref for --latest: release tag, else newest tag, else branch."""
    config = SNAPSHOT[library]
    if library == 'mpas':
        return config['ref']
    try:
        release = api_json(f"https://api.github.com/repos/{config['repo']}/releases/latest")
        if not release.get('draft') and not release.get('prerelease'):
            return release['tag_name']
    except Exception:
        pass
    tags = api_json(f"https://api.github.com/repos/{config['repo']}/tags?per_page=1")
    if tags:
        return tags[0]['name']
    return config['ref']


def checkout(stage, library, ref):
    repo = stage / (library + '-repo')
    subprocess.run(['git', 'init', str(repo)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin',
                    'https://github.com/' + SNAPSHOT[library]['repo'] + '.git'], check=True)
    subprocess.run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', ref], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    return repo


def copy_docs(source_dir, target_dir, exts, skip_dirs=('_build', 'venv', '.git', '__pycache__', '_static', '_templates')):
    """Copy matching-extension (or README-named) files, preserving the relative tree."""
    copied = 0
    for path in sorted(source_dir.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        if any(part in skip_dirs for part in relative.parts):
            continue
        if path.suffix.lower() not in exts and path.name.lower() != 'readme':
            continue
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def copy_licenses(repo, target_dir, names):
    """Copy named license/attribution files beside the corpus source (not indexed)."""
    copied = []
    for name in names:
        file = repo / name
        if file.is_file():
            shutil.copy2(file, target_dir / name)
            copied.append(name)
    return copied


def fetch_license_via_api(repo, commit, name, destination):
    """Fetch one repository file through the GitHub contents API.

    Used for crawled-site corpora (MOM6) where no checkout exists; raw.githubusercontent
    does not follow the repository's LICENSE symlink, the contents API does.
    """
    data = api_json(f'https://api.github.com/repos/{repo}/contents/{name}?ref={commit}')
    if data.get('encoding') != 'base64' or 'content' not in data:
        raise ValueError(f'{repo}:{name} is not a fetchable file via the contents API')
    destination.write_bytes(base64.b64decode(data['content']))
    return destination


def snapshot_entry(library, config, commit, ref):
    doc_dir = config['doc_dir']
    base = f"https://github.com/{config['repo']}/blob/{commit}"
    url = base if doc_dir == '.' else f"{base}/{doc_dir}"
    entry = {'name': library + '-docs', 'path': f'corpus/{library}/source', 'library': library,
             'version': config['version'], 'kind': 'documentation', 'managed': True, 'revision': commit,
             'url': url, 'version_basis': config['version_basis']}
    if 'target_release' in config:
        entry['target_release'] = config['target_release']
    return entry


def already_acquired(library, sources, existing_lock):
    """True when a library's corpus, lock record and catalog entry are all present.

    A refresh then reuses the existing acquisition instead of re-fetching. For
    the crawled MOM6 site this avoids re-tripping the Read the Docs throttle on
    every build; only an explicit ``--latest`` re-acquires a pinned snapshot.
    """
    if library not in existing_lock:
        return False
    if not any(entry.get('library') == library for entry in sources):
        return False
    source_dir = ROOT / 'corpus' / library / 'source'
    return source_dir.is_dir() and any(source_dir.iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'corpus.catalog.json')
    parser.add_argument('--only', help='Comma-separated subset of libraries to refresh')
    parser.add_argument('--latest', action='store_true',
                        help='Refresh snapshots to the newest release tag/branch and record new commits')
    args = parser.parse_args()

    wanted = {name.strip() for name in args.only.split(',')} if args.only else set(SNAPSHOT) | set(MOM6_SITES)
    unknown = wanted - set(SNAPSHOT) - set(MOM6_SITES)
    if unknown:
        parser.error(f'Unknown libraries: {", ".join(sorted(unknown))}')

    catalog_path = args.catalog
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {'sources': []}
    sources = catalog.get('sources', [])
    for entry in sources:
        if entry.get('library') in wanted and not entry.get('managed'):
            parser.error(f'{entry["library"]} has custom source entries; preserve them before refreshing')

    lock_path = ROOT / 'weather-lock.json'
    existing_lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    lock = {}
    skipped = set()
    with tempfile.TemporaryDirectory(prefix='weather-fetch-', dir=ROOT) as temp:
        stage = Path(temp)
        new_entries = []

        for library in SNAPSHOT:
            if library not in wanted:
                continue
            if not args.latest and already_acquired(library, sources, existing_lock):
                print(f'{library}: already acquired; skipping refresh', flush=True)
                skipped.add(library)
                continue
            config = dict(SNAPSHOT[library])
            ref = latest_ref(library) if args.latest else config['ref']
            repo = checkout(stage, library, ref)
            commit = git(repo, 'rev-parse', 'HEAD')
            if not args.latest and commit != config['commit']:
                raise ValueError(f'{library}: checkout does not match pinned commit')
            if git(repo, 'status', '--porcelain'):
                raise ValueError(f'{library}: checkout is modified')
            if args.latest:
                config['ref'] = ref
                config['commit'] = commit
                if library == 'mpas':
                    config['version'] = 'snapshot-' + commit[:12]
                else:
                    config['version'] = tag_version(library, ref)
            target = stage / library
            source = repo / config['doc_dir']
            if not source.is_dir():
                raise ValueError(f'{library}: docs directory {config["doc_dir"]} missing')
            count = copy_docs(source, target / 'source', tuple(e.lower() for e in config['exts']))
            if count == 0:
                raise ValueError(f'{library}: no documentation files found')
            copy_licenses(repo, target, config['licenses'])
            new_entries.append(snapshot_entry(library, config, commit, ref))
            examples = config.get('examples')
            if examples:
                example_source = repo / examples['doc_dir']
                if example_source.is_dir():
                    example_count = copy_docs(example_source, target / 'examples',
                                              tuple(e.lower() for e in examples['exts']))
                    if example_count:
                        new_entries.append({
                            'name': library + '-examples', 'path': f'corpus/{library}/examples',
                            'library': library, 'version': config['version'], 'kind': 'example',
                            'managed': True, 'revision': commit,
                            'url': f"https://github.com/{config['repo']}/blob/{commit}/{examples['doc_dir']}",
                            'version_basis': examples['version_basis']})
            lock[library] = {**config, 'exts': list(config['exts']),
                             'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}

        for library in MOM6_SITES:
            if library not in wanted:
                continue
            if not args.latest and already_acquired(library, sources, existing_lock):
                print(f'{library}: already acquired; skipping refresh', flush=True)
                skipped.add(library)
                continue
            config = MOM6_SITES[library]
            # Prefer the site's single htmlzip artifact: one request instead of a
            # page-by-page crawl, which avoids tripping the Read the Docs throttle.
            # Fall back to a patient crawl if the artifact endpoint is unavailable.
            try:
                # Wait out a persistent throttle window: up to 10 attempts, each
                # backing off to at most 15 minutes, all as single requests.
                stored = download_html_zip(config['htmlzip_url'], stage / library / 'source',
                                           retries=10, max_delay=900.0)
                count = len(stored)
                source_kind = 'htmlzip'
            except Exception as error:
                print(f'{library}: htmlzip download failed ({error}); crawling the site instead',
                      flush=True)
                # Read the Docs throttles aggressive crawlers; pace every request
                # with a few workers and a patient per-page retry so the rate
                # budget is respected instead of tripped.
                count = crawl_site(library, stage / library / 'source', sites=MOM6_SITES,
                                   workers=2, retries=12, delay=2.0)
                source_kind = 'crawl'
            if count == 0:
                raise ValueError(f'{library}: no pages were fetched')
            # The crawled site carries no license; keep the repository's Apache-2.0
            # LICENSE (a symlink raw URLs do not serve) beside the corpus.
            fetch_license_via_api('mom-ocean/MOM6', config['revision'], 'LICENSE',
                                  stage / library / 'LICENSE')
            new_entries.append({'name': library + '-docs', 'path': config['path'], 'library': library,
                                'version': config['version'], 'kind': 'documentation', 'managed': True,
                                'revision': config['revision'], 'url': config['base_url'],
                                'version_basis': config['version_basis']})
            lock[library] = {**config, 'pages': count, 'acquisition': source_kind,
                             'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}

        # No live corpus changes occur until every source has been verified and staged.
        for library in wanted:
            destination = ROOT / 'corpus' / library
            if (stage / library).exists():
                if destination.exists():
                    shutil.rmtree(destination)
                destination.mkdir(parents=True, exist_ok=True)
                for child in (stage / library).iterdir():
                    shutil.move(str(child), str(destination / child.name))
        refreshed = wanted - skipped
        sources = [entry for entry in sources if entry.get('library') not in refreshed]
        sources.extend(new_entries)
        catalog_path.write_text(json.dumps({'sources': sources}, indent=2) + '\n')
        existing_lock.update(lock)
        lock_path.write_text(json.dumps(existing_lock, indent=2) + '\n')
    imported = sorted(wanted - skipped)
    if imported:
        print(f'Imported weather/workflow collections: {", ".join(imported)}. Run ingest.py to rebuild')
    if skipped:
        print(f'Reused existing acquisitions: {", ".join(sorted(skipped))}')


if __name__ == '__main__':
    main()
