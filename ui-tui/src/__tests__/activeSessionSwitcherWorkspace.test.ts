import { describe, expect, it } from 'vitest'

import {
  orchestratorGlobalHotkeyHint,
  orchestratorGlobalHotkeyHintSegments,
  scopeLabel,
  sessionListParams
} from '../components/activeSessionSwitcher.js'
import { messages } from '../i18n/runtime.js'

describe('session switcher workspace scope', () => {
  it('sends workspace_cwd only when scoped to a known project', () => {
    expect(sessionListParams('/repo/app', false)).toEqual({ limit: 200, workspace_cwd: '/repo/app' })
    expect(sessionListParams('/repo/app', true)).toEqual({ limit: 200 })
    expect(sessionListParams(undefined, false)).toEqual({ limit: 200 })
    expect(sessionListParams('', false)).toEqual({ limit: 200 })
  })

  it('labels a filtered list with the project folder name', () => {
    expect(scopeLabel('/Users/me/Git/hermes')).toBe(messages().pickers.session.scope.project('hermes'))
    expect(scopeLabel('/Users/me/Git/hermes/')).toBe('this project: hermes')
  })

  it('labels an unfiltered list as all projects', () => {
    expect(scopeLabel(null)).toBe(messages().pickers.session.scope.all)
    expect(scopeLabel(undefined)).toBe('all projects')
  })

  it('advertises Ctrl+A in the global hotkey hint', () => {
    expect(orchestratorGlobalHotkeyHintSegments()).toContainEqual({ role: 'hotkey', text: 'Ctrl+A' })
    expect(orchestratorGlobalHotkeyHint()).toContain('Ctrl+A all/project · ')
  })
})
