document$.subscribe(function() {
  // Only run on api-completeness pages
  var currentPath = window.location.pathname
  if (!currentPath.includes('/api-completeness/')) {
    return
  }
  
  var tables = document.querySelectorAll("article table:not([class])")
  tables.forEach(function(table) {
    new Tablesort(table)
    addColumnSelector(table)
    addMethodFilter(table)
  })
})

function addColumnSelector(table) {
  var thead = table.querySelector('thead')
  var headerRow = thead.querySelector('tr')
  var headers = Array.from(headerRow.querySelectorAll('th'))
  
  // Skip if table only has one column (Method column)
  if (headers.length <= 1) return
  
  // Get backend columns (all except first "Method" column)
  var backendColumns = headers.slice(1).map(function(th, index) {
    return {
      name: th.textContent.trim(),
      index: index + 1,
      visible: true
    }
  })
  
  // Sort backends alphabetically
  backendColumns.sort(function(a, b) {
    return a.name.localeCompare(b.name)
  })
  
  // Create container for the dropdown
  var container = document.createElement('div')
  container.className = 'column-selector-container'
  container.style.cssText = 'margin: 16px 0; position: relative; display: inline-block;'
  
  // Create the dropdown button
  var button = document.createElement('button')
  button.className = 'column-selector-button'
  button.textContent = 'Select Backends'
  button.style.cssText = `
    padding: 8px 16px;
    background: var(--md-primary-fg-color, #4051b5);
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s;
  `
  
  // Add arrow icon
  var arrow = document.createElement('span')
  arrow.textContent = '▼'
  arrow.style.cssText = 'font-size: 10px; transition: transform 0.2s;'
  button.appendChild(arrow)
  
  // Create dropdown menu
  var dropdown = document.createElement('div')
  dropdown.className = 'column-selector-dropdown'
  dropdown.style.cssText = `
    position: absolute;
    top: 100%;
    left: 0;
    margin-top: 4px;
    background: var(--md-default-bg-color, white);
    border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    min-width: 200px;
    max-height: 300px;
    overflow-y: auto;
    display: none;
    z-index: 1000;
  `
  
  // Add "Select All" / "Deselect All" buttons
  var selectAllContainer = document.createElement('div')
  selectAllContainer.style.cssText = `
    padding: 8px 12px;
    border-bottom: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
    display: flex;
    gap: 8px;
  `
  
  var selectAllBtn = document.createElement('button')
  selectAllBtn.textContent = 'Select All'
  selectAllBtn.style.cssText = `
    flex: 1;
    padding: 4px 8px;
    font-size: 12px;
    border: 1px solid var(--md-default-fg-color--lighter, #ccc);
    border-radius: 4px;
    background: var(--md-default-bg-color, white);
    color: var(--md-default-fg-color, black);
    cursor: pointer;
    transition: background 0.2s;
  `
  
  var deselectAllBtn = document.createElement('button')
  deselectAllBtn.textContent = 'Clear All'
  deselectAllBtn.style.cssText = selectAllBtn.style.cssText
  
  selectAllContainer.appendChild(selectAllBtn)
  selectAllContainer.appendChild(deselectAllBtn)
  dropdown.appendChild(selectAllContainer)
  
  // Create checkboxes for each backend
  backendColumns.forEach(function(column) {
    var label = document.createElement('label')
    label.style.cssText = `
      display: flex;
      align-items: center;
      padding: 8px 12px;
      cursor: pointer;
      transition: background 0.2s;
      user-select: none;
    `
    label.addEventListener('mouseenter', function() {
      label.style.background = 'var(--md-default-fg-color--lightest, #f5f5f5)'
    })
    label.addEventListener('mouseleave', function() {
      label.style.background = 'transparent'
    })
    
    var checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.checked = true
    checkbox.style.cssText = 'margin-right: 8px; cursor: pointer;'
    checkbox.dataset.columnIndex = column.index
    
    var span = document.createElement('span')
    span.textContent = column.name
    span.style.cssText = 'font-size: 14px; color: var(--md-default-fg-color, black);'
    
    label.appendChild(checkbox)
    label.appendChild(span)
    dropdown.appendChild(label)
    
    // Handle column visibility toggle
    checkbox.addEventListener('change', function() {
      toggleColumn(table, column.index, checkbox.checked)
      updateButtonText()
    })
  })
  
  // Toggle dropdown visibility
  button.addEventListener('click', function(e) {
    e.stopPropagation()
    var isVisible = dropdown.style.display === 'block'
    dropdown.style.display = isVisible ? 'none' : 'block'
    arrow.style.transform = isVisible ? 'rotate(0deg)' : 'rotate(180deg)'
  })
  
  // Close dropdown when clicking outside
  document.addEventListener('click', function(e) {
    if (!container.contains(e.target)) {
      dropdown.style.display = 'none'
      arrow.style.transform = 'rotate(0deg)'
    }
  })
  
  // Select all handler
  selectAllBtn.addEventListener('click', function(e) {
    e.stopPropagation()
    var checkboxes = dropdown.querySelectorAll('input[type="checkbox"]')
    checkboxes.forEach(function(cb) {
      cb.checked = true
      toggleColumn(table, parseInt(cb.dataset.columnIndex), true)
    })
    updateButtonText()
  })
  
  // Deselect all handler
  deselectAllBtn.addEventListener('click', function(e) {
    e.stopPropagation()
    var checkboxes = dropdown.querySelectorAll('input[type="checkbox"]')
    checkboxes.forEach(function(cb) {
      cb.checked = false
      toggleColumn(table, parseInt(cb.dataset.columnIndex), false)
    })
    updateButtonText()
  })
  
  function updateButtonText() {
    var checkboxes = Array.from(dropdown.querySelectorAll('input[type="checkbox"]'))
    var checkedCount = checkboxes.filter(function(cb) { return cb.checked }).length
    var totalCount = checkboxes.length
    
    if (checkedCount === 0) {
      button.firstChild.textContent = 'Select Backends'
    } else if (checkedCount === totalCount) {
      button.firstChild.textContent = 'All Backends Selected'
    } else {
      button.firstChild.textContent = checkedCount + ' of ' + totalCount + ' Selected'
    }
  }
  
  container.appendChild(button)
  container.appendChild(dropdown)
  
  // Wrap table in a container for better layout control
  var tableContainer = document.createElement('div')
  tableContainer.className = 'table-container'
  table.parentNode.insertBefore(tableContainer, table)
  tableContainer.appendChild(table)
  
  // Insert button container before the table container
  tableContainer.parentNode.insertBefore(container, tableContainer)
}

function toggleColumn(table, columnIndex, visible) {
  var rows = table.querySelectorAll('tr')
  rows.forEach(function(row) {
    var cells = row.querySelectorAll('th, td')
    if (cells[columnIndex]) {
      cells[columnIndex].style.display = visible ? '' : 'none'
    }
  })
}

function addMethodFilter(table) {
  var thead = table.querySelector('thead')
  var headerRow = thead.querySelector('tr')
  var headers = headerRow.querySelectorAll('th')
  
  // Create filter input row
  var filterRow = document.createElement('tr')
  filterRow.className = 'filter-row'
  
  headers.forEach(function(header, index) {
    var th = document.createElement('th')
    
    // Add input only to first column (Method column)
    if (index === 0) {
      var input = document.createElement('input')
      input.type = 'text'
      input.placeholder = 'Filter methods...'
      input.style.width = '100%'
      input.style.padding = '4px'
      input.style.boxSizing = 'border-box'
      input.style.border = '1px solid #ccc'
      input.style.borderRadius = '4px'
      
      input.addEventListener('input', function(e) {
        filterTable(table, e.target.value)
      })
      
      th.appendChild(input)
    }
    
    filterRow.appendChild(th)
  })
  
  thead.appendChild(filterRow)
}

function filterTable(table, filterValue) {
  var tbody = table.querySelector('tbody')
  var rows = tbody.querySelectorAll('tr')
  var lowerFilterValue = filterValue.toLowerCase()
  
  rows.forEach(function(row) {
    var firstCell = row.querySelector('td')
    var cellText = firstCell.textContent.toLowerCase()
    var originalText = firstCell.textContent
    
    if (cellText.includes(lowerFilterValue)) {
      row.style.display = ''
      
      // Highlight matching text
      if (lowerFilterValue) {
        var regex = new RegExp('(' + escapeRegex(filterValue) + ')', 'gi')
        firstCell.innerHTML = originalText.replace(regex, '<mark style="background-color: yellow; padding: 0;">$1</mark>')
      } else {
        firstCell.textContent = originalText
      }
    } else {
      row.style.display = 'none'
    }
  })
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
