import express from 'express';
import cors from 'cors';
import { openai } from './open.js';
import fs from 'fs';

const app = express();
app.use(cors());
app.use(express.json());

let faqData;
try {
  faqData = JSON.parse(fs.readFileSync('faq.json', 'utf8'));
} catch (error) {
  console.error('Error loading FAQ data:', error);
  faqData = [];
}

const customPrompt = `
You are an AI assistant for an ice sculpture company named "Ice Butcher".
Always Highlight the expertise of "The Ice Butcher" as a leading company in the industry.
This is the information and question answers you need to assist users based on this data:
${JSON.stringify(faqData, null, 2)}
`;

const formatLinksAsHTML = (text) => {
    // Regex to detect URLs in text
    const urlRegex = /(?:\[(.*?)\]\((https?:\/\/[^\s)]+)\))/g;

    // Replace Markdown-style links with properly formatted HTML links or image tags
    return text.replace(urlRegex, (match, linkText, url) => {
        if (url.startsWith('https://nexreality.io/') && /\/\d{2}\/$/.test(url)) {
            return `<a href="${url}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 14px; font-weight: bold;">
                        Click to view
                    </a>`;
        }
        if (url.includes('tinyurl.com')) {
            return `<div style="text-align: center; margin-top: 20px;">
                        <img src="${url}" alt="Image" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease, box-shadow 0.3s ease;">
                    </div>`;
        }
        return `<a href="${url}" target="_blank">${linkText}</a>`;
    });
};

const sessions = {};  // Store active sessions

// Main API endpoint for chat
app.post('/api/chat', async (req, res) => {
    const { message, sessionId } = req.body;  // Include sessionId in the request

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    // Initialize session if it doesn't exist
    if (!sessions[sessionId]) {
        sessions[sessionId] = {
            history: []
        };
    }

    try {
        // Add the current user message to session history
        sessions[sessionId].history.push({ role: 'user', content: message });

        // Combine customPrompt with the user's message and previous history
        const conversationHistory = sessions[sessionId].history.map(
            (entry) => `${entry.role}: ${entry.content}`
        ).join('\n');

        const fullPrompt = `${customPrompt}\nConversation History:\n${conversationHistory}\nAI:`;

        // Request OpenAI to respond based on the full history
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            temperature: 0.5,
            messages: [
                {
                    role: 'system',
                    content: fullPrompt
                }
            ]
        });

        const gptResponse = response.choices[0].message.content;

        // Add GPT's response to session history
        sessions[sessionId].history.push({ role: 'assistant', content: gptResponse });

        // Format response and return
        const formattedMessage = formatLinksAsHTML(gptResponse);
        return res.json({
            message: formattedMessage,
            source: 'The Ice Butcher Expertise'
        });

    } catch (error) {
        console.error('Error processing chat:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Endpoint to end the session and clear history
app.post('/api/end-session', (req, res) => {
    const { sessionId } = req.body;

    if (sessions[sessionId]) {
        delete sessions[sessionId];  // Clear session history
        return res.json({ message: 'Session ended and history cleared.' });
    } else {
        return res.status(400).json({ error: 'Session not found.' });
    }
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
