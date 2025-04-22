/**
 * Henry George Chatbot - JavaScript Functionality
 */

const API_URL = 'http://localhost:5000/api';

// Global variable to track selected documents
let selectedDocuments = [];

// Wait for the DOM to be fully loaded before executing scripts
document.addEventListener('DOMContentLoaded', () => {
    // ===== INITIALIZE AOS ANIMATIONS =====
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: false,
        mirror: false
    });
    
    // ===== THEME TOGGLE FUNCTIONALITY =====
    initThemeToggle();
    
    // ===== NAVIGATION FUNCTIONALITY =====
    initNavigation();
    
    // ===== CHAT INTERFACE FUNCTIONALITY =====
    initChatInterface();
    
    // ===== DOCUMENT UPLOAD FUNCTIONALITY =====
    initDocumentUpload();
    
    // ===== DOCUMENT SELECTION FUNCTIONALITY =====
    initDocumentSelection();
    
    // ===== CITATIONS FUNCTIONALITY =====
    initCitationsDisplay();
    
    // ===== FADE-IN ANIMATIONS =====
    initFadeInAnimations();
    
    // ===== FETCH AVAILABLE DOCUMENTS =====
    fetchAvailableDocuments();
    
    // ===== MISC FUNCTIONALITY =====
    // Set current year in footer
    document.getElementById('current-year').textContent = new Date().getFullYear();
    
    // Auto-resize textarea
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', autoResizeTextarea);
    }
});

/**
 * Function to get selected documents
 * Returns an array of document IDs that are currently selected
 */
function getSelectedDocuments() {
    const documentItems = document.querySelectorAll('.document-item.active');
    // Make sure we're getting the exact ID as stored in the database
    selectedDocuments = Array.from(documentItems).map(item => item.getAttribute('data-doc-id'));
    console.log('Selected documents:', selectedDocuments);
    return selectedDocuments;
}

/**
 * Function to fetch and display available documents from the server
 */
async function fetchAvailableDocuments() {
    try {
        console.log('Fetching available documents...');
        const response = await fetch(`${API_URL}/documents`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Documents received:', data);
        
        const documentList = document.getElementById('document-list');
        
        if (!documentList) {
            console.error('Document list element not found');
            return;
        }
        
        // Clear existing list except default items
        const defaultItems = Array.from(documentList.querySelectorAll('.document-item:not([data-uploaded="true"])'));
        
        // Add new items from server
        data.documents.forEach(doc => {
            // Skip if it's already in the default list
            if (defaultItems.some(item => item.getAttribute('data-doc-id') === doc.id)) {
                return;
            }
            
            const docItem = document.createElement('li');
            docItem.className = 'document-item';
            docItem.setAttribute('data-doc-id', doc.id);
            docItem.setAttribute('data-uploaded', 'true');
            docItem.innerHTML = `<span class="doc-title">${doc.name}</span>`;
            
            // Add event listener to select document
            docItem.addEventListener('click', () => {
                selectDocument(docItem);
            });
            
            documentList.appendChild(docItem);
        });
        
        // Select the first document by default if nothing is selected
        setTimeout(() => {
            if (getSelectedDocuments().length === 0) {
                const firstDoc = document.querySelector('.document-item');
                if (firstDoc) {
                    selectDocument(firstDoc);
                }
            }
        }, 500);
        
    } catch (error) {
        console.error('Error fetching documents:', error);
    }
}

/**
 * Initialize chat interface functionality
 * Handles sending and receiving messages
 */
function initChatInterface() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const clearChatButton = document.getElementById('clear-chat');
    
    // Function to add a user message to the chat
    const addUserMessage = (message) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    };
    
    // Function to add an AI message to the chat
    const addAIMessage = (message, citations = [], followUps = []) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';
        
        // Create message content
        let messageHTML = `
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
        
        // Add citations if available
        if (citations && citations.length > 0) {
            messageHTML += `
                <div class="message-citations">
                    <h4>Citations:</h4>
                    <ul>
                        ${citations.map(citation => `
                            <li>
                                <span class="citation-source">${citation.book || ''}</span> - 
                                ${citation.chapter || ''}: ${citation.chapter_title || ''}
                            </li>
                        `).join('')}
                    </ul>
                    <button class="expand-citation">View Excerpt</button>
                    <div class="citation-excerpt hidden">
                        <blockquote>
                            ${citations[0].excerpt || 'No excerpt available'}
                        </blockquote>
                    </div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageHTML;
        chatMessages.appendChild(messageDiv);
        
        // Add follow-up questions if available
        if (followUps && followUps.length > 0) {
            updateFollowUpQuestions(followUps);
        }
        
        scrollToBottom();
    };
    
    // Function to handle chat form submission
    const handleChatSubmit = (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();

        if (message) {
            // Add user message to chat
            addUserMessage(message);

            // Clear input field
            chatInput.value = '';

            // Reset textarea height
            chatInput.style.height = 'auto';

            // Show typing indicator
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'message ai-message typing';
            typingIndicator.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            chatMessages.appendChild(typingIndicator);
            scrollToBottom();

            // Get selected documents
            const docs = getSelectedDocuments();
            console.log('Sending query with documents:', docs);

            // Send query to API
            fetch(`${API_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    namespaces: docs
                }),
            })
            .then(response => {
                // Check if the response is ok
                if (!response.ok) {
                    console.error(`HTTP error! Status: ${response.status}`);
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Response received:', data);
                
                // Remove typing indicator
                if (typingIndicator.parentNode) {
                    chatMessages.removeChild(typingIndicator);
                }

                // Add AI response to chat
                addAIMessage(
                    data.response, 
                    data.structured_response.citations, 
                    data.structured_response.follow_up_questions
                );
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Remove typing indicator
                if (typingIndicator.parentNode) {
                    chatMessages.removeChild(typingIndicator);
                }

                // Add error message
                addAIMessage("I'm sorry, I encountered an error while processing your question. Please try again.");
            });
        }
    };
    
    // Function to update follow-up questions
    const updateFollowUpQuestions = (questions) => {
        const followUpContainer = document.getElementById('follow-up-container');
        if (!followUpContainer) return;
        
        const followUpButtons = followUpContainer.querySelector('.follow-up-buttons');
        if (!followUpButtons) return;
        
        // Clear existing buttons
        followUpButtons.innerHTML = '';
        
        // Add new buttons
        questions.forEach(question => {
            const button = document.createElement('button');
            button.className = 'follow-up-button';
            button.textContent = question;
            button.addEventListener('click', () => {
                chatInput.value = question;
                handleChatSubmit(new Event('submit'));
            });
            followUpButtons.appendChild(button);
        });
        
        // Show follow-up container
        followUpContainer.style.display = 'block';
    };
    
    // Function to clear chat history
    const clearChat = () => {
        // Keep only the welcome message
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Hide follow-up questions
        const followUpContainer = document.getElementById('follow-up-container');
        if (followUpContainer) {
            followUpContainer.style.display = 'none';
        }
    };
    
    // Add event listeners
    if (chatForm) {
        chatForm.addEventListener('submit', handleChatSubmit);
    }
    
    if (clearChatButton) {
        clearChatButton.addEventListener('click', clearChat);
    }
    
    // Make addUserMessage and addAIMessage available globally
    window.addUserMessage = addUserMessage;
    window.addAIMessage = addAIMessage;
    window.handleChatSubmit = handleChatSubmit;
}

/**
 * Initialize document upload functionality
 * Handles file uploads and preview
 */
function initDocumentUpload() {
    const documentUploadForm = document.getElementById('document-upload-form');
    const fileInput = document.getElementById('document-upload');
    const uploadPreview = document.getElementById('upload-preview');
    const fileDropArea = document.querySelector('.file-drop-area');
    const uploadMiniButton = document.getElementById('upload-mini-button');
    
    // Array to store uploaded files
    let uploadedFiles = [];
    
    // Function to handle file selection
    const handleFileSelect = (e) => {
        const files = e.target.files || e.dataTransfer.files;
        
        if (files.length > 0) {
            // Add files to uploaded files array
            for (let i = 0; i < files.length; i++) {
                if (!isFileAlreadyUploaded(files[i].name)) {
                    uploadedFiles.push(files[i]);
                }
            }
            
            // Update preview
            updateFilePreview();
        }
    };
    
    // Function to check if file is already uploaded
    const isFileAlreadyUploaded = (fileName) => {
        return uploadedFiles.some(file => file.name === fileName);
    };
    
    // Function to update file preview
    const updateFilePreview = () => {
        if (!uploadPreview) return;
        
        uploadPreview.innerHTML = '';
        
        uploadedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            // Determine file type icon
            let fileIcon;
            if (file.name.endsWith('.pdf')) {
                fileIcon = 'fa-file-pdf';
            } else if (file.name.endsWith('.docx') || file.name.endsWith('.doc')) {
                fileIcon = 'fa-file-word';
            } else if (file.name.endsWith('.txt')) {
                fileIcon = 'fa-file-alt';
            } else {
                fileIcon = 'fa-file';
            }
            
            fileItem.innerHTML = `
                <i class="fas ${fileIcon}"></i>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
                <button class="remove-file" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            uploadPreview.appendChild(fileItem);
        });
        
        // Add event listeners to remove buttons
        const removeButtons = document.querySelectorAll('.remove-file');
        removeButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.getAttribute('data-index'));
                uploadedFiles.splice(index, 1);
                updateFilePreview();
            });
        });
    };
    
    // Function to format file size
    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    // Handle document upload
    const handleDocumentUpload = async (e) => {
        e.preventDefault();
        
        if (uploadedFiles.length === 0) {
            alert('Please select at least one file to upload.');
            return;
        }
        
        // Show loading indicator
        const uploadButton = e.target.querySelector('button[type="submit"]');
        const originalButtonText = uploadButton.innerHTML;
        uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        uploadButton.disabled = true;
        
        // Create form data for upload
        const formData = new FormData();
        formData.append('file', uploadedFiles[0]); // Process one file at a time
        
        try {
            console.log('Uploading file:', uploadedFiles[0].name);
            
            // Send upload request
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }
            
            const data = await response.json();
            console.log('Upload successful:', data);
            
            // Success message
            alert(`File processed successfully!`);
            
            // Clear uploaded files
            uploadedFiles = [];
            updateFilePreview();
            
            // Refresh available documents
            fetchAvailableDocuments();
            
            // Navigate to chat interface
            const chatLinks = document.querySelectorAll('a[href="#chat"]');
            if (chatLinks.length > 0) {
                chatLinks[0].click();
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            // Reset button
            uploadButton.innerHTML = originalButtonText;
            uploadButton.disabled = false;
        }
    };
    
    // Add event listeners for file drop area
    if (fileDropArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, () => {
                fileDropArea.classList.add('highlight');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, () => {
                fileDropArea.classList.remove('highlight');
            }, false);
        });
        
        fileDropArea.addEventListener('drop', handleFileSelect, false);
    }
    
    // Add event listener for file input
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Add event listener for upload mini button
    if (uploadMiniButton) {
        uploadMiniButton.addEventListener('click', () => {
            // Scroll to upload section
            document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
        });
    }
    
    // Add event listener for form submission
    if (documentUploadForm) {
        documentUploadForm.addEventListener('submit', handleDocumentUpload);
    }
    
    // Make uploadedFiles available globally
    window.uploadedFiles = uploadedFiles;
}

/**
 * Initialize document selection functionality
 * Handles selecting documents in the sidebar
 */
function initDocumentSelection() {
    const documentList = document.getElementById('document-list');
    const documentSearch = document.getElementById('document-search');
    
    // Function to select a document
    const selectDocument = (docItem) => {
        // Toggle active class
        docItem.classList.toggle('active');
        
        // Update selectedDocuments array
        selectedDocuments = getSelectedDocuments();
        
        console.log('Selected documents:', selectedDocuments);
    };
    
    // Add event listeners to document items
    if (documentList) {
        const documentItems = documentList.querySelectorAll('.document-item');
        documentItems.forEach(item => {
            item.addEventListener('click', () => {
                selectDocument(item);
            });
        });
    }
    
    // Add event listener for document search
    if (documentSearch) {
        documentSearch.addEventListener('input', () => {
            const searchTerm = documentSearch.value.toLowerCase();
            const documentItems = document.querySelectorAll('.document-item');
            
            documentItems.forEach(item => {
                const docTitle = item.querySelector('.doc-title').textContent.toLowerCase();
                
                if (docTitle.includes(searchTerm)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
    
    // Make selectDocument function globally available
    window.selectDocument = selectDocument;
}

/**
 * Initialize theme toggle functionality
 * Handles switching between light and dark mode
 */
function initThemeToggle() {
    // Get theme toggle buttons
    const mainThemeToggle = document.getElementById('theme-toggle');
    const chatThemeToggle = document.getElementById('chat-theme-toggle');
    
    // Check for saved theme preference or use default
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
    } else {
        document.body.removeAttribute('data-theme');
    }
    
    // Function to toggle theme
    const toggleTheme = () => {
        if (document.body.getAttribute('data-theme') === 'dark') {
            document.body.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
        } else {
            document.body.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        }
    };
    
    // Add event listeners to theme toggle buttons
    if (mainThemeToggle) {
        mainThemeToggle.addEventListener('click', toggleTheme);
    }
    
    if (chatThemeToggle) {
        chatThemeToggle.addEventListener('click', toggleTheme);
    }
}

/**
 * Initialize navigation functionality
 * Handles switching between landing page and chat interface
 */
function initNavigation() {
    // Get navigation elements
    const startChatButtons = document.querySelectorAll('a[href="#chat"], .cta-button[href="#chat"]');
    const backToLandingButton = document.getElementById('back-to-landing');
    const landingPage = document.getElementById('landing-page');
    const chatInterface = document.getElementById('chat');
    
    // Function to show chat interface and hide landing page
    const showChatInterface = (e) => {
        if (e) e.preventDefault();
        landingPage.classList.add('hidden');
        chatInterface.classList.remove('hidden');
        // Scroll to the bottom of chat messages
        scrollToBottom();
    };
    
    // Function to show landing page and hide chat interface
    const showLandingPage = (e) => {
        if (e) e.preventDefault();
        chatInterface.classList.add('hidden');
        landingPage.classList.remove('hidden');
        // Scroll to top of page
        window.scrollTo(0, 0);
    };
    
    // Add event listeners to navigation buttons
    startChatButtons.forEach(button => {
        button.addEventListener('click', showChatInterface);
    });
    
    if (backToLandingButton) {
        backToLandingButton.addEventListener('click', showLandingPage);
    }
    
    // Check if URL has #chat hash
    if (window.location.hash === '#chat') {
        showChatInterface();
    }
}

/**
 * Initialize citations display functionality
 * Handles expanding and collapsing citations
 */
function initCitationsDisplay() {
    // Event delegation for citation expand/collapse
    document.addEventListener('click', (e) => {
        if (e.target && e.target.classList.contains('expand-citation')) {
            const excerpt = e.target.nextElementSibling;
            excerpt.classList.toggle('hidden');
            e.target.textContent = excerpt.classList.contains('hidden') ? 'View Excerpt' : 'Hide Excerpt';
        }
    });
}

/**
 * Utility function to scroll chat to bottom
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

/**
 * Utility function to auto-resize textarea
 */
function autoResizeTextarea() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
}

/**
 * Initialize fade-in animations for elements with the fade-in class
 * Uses Intersection Observer API for scroll-based animations
 */
function initFadeInAnimations() {
    const fadeElements = document.querySelectorAll('.fade-in');
    
    if (fadeElements.length > 0) {
        const fadeObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                } else {
                    // Uncomment the line below if you want elements to fade out when scrolled away
                    // entry.target.classList.remove('visible');
                }
            });
        }, {
            threshold: 0.2
        });
        
        fadeElements.forEach(element => {
            fadeObserver.observe(element);
        });
    }
}