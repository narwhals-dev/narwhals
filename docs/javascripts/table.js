document$.subscribe(function() {
  var tables = document.querySelectorAll("article table:not([class])")
  tables.forEach(function(table) {
    new Tablesort(table)
    addMethodFilter(table)
  })
})

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
