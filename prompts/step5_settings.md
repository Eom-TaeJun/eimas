# Step 5: EIMAS Settings Page

Create settings/configuration page.

## Route: /settings

### Components

1. **Account Settings**
   - Account name input (default: "default")
   - Initial cash balance input

2. **Analysis Settings**
   - Default analysis level dropdown
   - Default research goal dropdown
   - Skip stages checkboxes (data/analysis/debate)

3. **Data Refresh Settings**
   - Auto-refresh interval radio buttons (30/60/120 seconds)
   - Toggle auto-refresh on/off

4. **Theme Settings**
   - Light/Dark mode toggle switch
   - Save button

### Storage
- Use localStorage for settings persistence
- No backend API needed

### Tech Stack
- shadcn/ui: Input, Select, Checkbox, Switch, Button, Card
- React useState for settings
- next-themes for theme switching

Generate with proper form state management and localStorage persistence.
