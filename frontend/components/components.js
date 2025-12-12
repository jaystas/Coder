// AI Chat Interface - Component Library JavaScript
// ================================================

// Initialize all components when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initCustomDropdowns();
    initTabs();
});

// Custom Dropdown Component
function initCustomDropdowns() {
    const dropdowns = document.querySelectorAll('.dropdown');
    
    dropdowns.forEach(dropdown => {
        const trigger = dropdown.querySelector('.dropdown-trigger');
        const menu = dropdown.querySelector('.dropdown-menu');
        const items = dropdown.querySelectorAll('.dropdown-item');
        const valueDisplay = dropdown.querySelector('.dropdown-value');
        
        if (!trigger || !menu) return;
        
        // Toggle dropdown
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            
            // Close other dropdowns
            dropdowns.forEach(other => {
                if (other !== dropdown) {
                    other.classList.remove('active');
                }
            });
            
            dropdown.classList.toggle('active');
        });
        
        // Handle item selection
        items.forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                
                // Remove selected class from all items
                items.forEach(i => i.classList.remove('selected'));
                
                // Add selected class to clicked item
                item.classList.add('selected');
                
                // Update display value
                const title = item.querySelector('.dropdown-item-title');
                if (valueDisplay && title) {
                    valueDisplay.textContent = title.textContent;
                }
                
                // Close dropdown
                dropdown.classList.remove('active');
                
                // Trigger custom event
                const value = item.getAttribute('data-value');
                dropdown.dispatchEvent(new CustomEvent('dropdown-change', {
                    detail: { value, element: item }
                }));
            });
        });
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
        dropdowns.forEach(dropdown => {
            dropdown.classList.remove('active');
        });
    });
}

// Tabs Component
function initTabs() {
    const tabsContainers = document.querySelectorAll('.tabs');
    
    tabsContainers.forEach(container => {
        const buttons = container.querySelectorAll('.tab-button');
        const panels = container.querySelectorAll('.tab-panel');
        
        buttons.forEach(button => {
            button.addEventListener('click', () => {
                const targetId = button.getAttribute('data-tab');
                
                // Remove active class from all buttons and panels
                buttons.forEach(btn => btn.classList.remove('active'));
                panels.forEach(panel => panel.classList.remove('active'));
                
                // Add active class to clicked button and corresponding panel
                button.classList.add('active');
                const targetPanel = container.querySelector(`#${targetId}`);
                if (targetPanel) {
                    targetPanel.classList.add('active');
                }
                
                // Trigger custom event
                container.dispatchEvent(new CustomEvent('tab-change', {
                    detail: { tabId: targetId }
                }));
            });
        });
    });
}
