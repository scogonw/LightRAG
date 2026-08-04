import { useCallback } from 'react'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useTranslation } from 'react-i18next'

import { SearchIcon, XIcon } from 'lucide-react'

interface DocumentSearchBarProps {
  value: string
  onValueChange: (value: string) => void
  onClear: () => void
}

/**
 * Controlled file name search input for the documents list.
 *
 * Presentational only — debouncing, request building and reset behavior all
 * live in DocumentManager, which owns the state.
 */
const DocumentSearchBar = ({ value, onValueChange, onClear }: DocumentSearchBarProps) => {
  const { t } = useTranslation()

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Escape' && value.length > 0) {
        event.preventDefault()
        onClear()
      }
    },
    [onClear, value]
  )

  return (
    <div className="relative flex items-center">
      <SearchIcon className="text-muted-foreground pointer-events-none absolute start-2.5 h-4 w-4" />
      <Input
        type="text"
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={t('documentPanel.documentManager.search.placeholder')}
        aria-label={t('documentPanel.documentManager.search.ariaLabel')}
        className="w-64 ps-8 pe-8"
      />
      {value.length > 0 && (
        <Button
          variant="ghost"
          size="icon"
          onClick={onClear}
          className="absolute end-0.5 h-7 w-7"
          tooltip={t('documentPanel.documentManager.search.clear')}
        >
          <XIcon className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}

export default DocumentSearchBar
