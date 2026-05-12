# ChaiIntel Theme (Tailwind v4 + TailAdmin)

This folder contains the source CSS for the ChaiIntel dashboard.
The visual style is based on [TailAdmin](https://tailadmin.com/) (MIT licence).

## Build

```bash
cd theme
npm install
npm run build       # one-shot, minified -> ../analytics/static/analytics/theme/style.css
npm run watch       # rebuild on save during development
```

The compiled CSS is committed to `analytics/static/analytics/theme/style.css`
so that the Django app runs without a Node toolchain in production.

## How class scanning works

Tailwind v4 scans the files listed via `@source` directives in `src/style.css`.
We point it at all Django template files so any utility class used in a `.html`
template will be emitted.
