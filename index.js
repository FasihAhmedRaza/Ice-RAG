// server.js
import express from 'express';
import cors from 'cors';
import { openai } from './open.js';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

const app = express();
app.use(cors());
app.use(express.json());

const pdf = './faqs.pdf';
let vectorStore = null;

const customPrompt = {
    temperature: 0.5, // Balanced approach for accuracy and creativity
    systemPrompt: `You are an expert in ice sculptures and ice butchery with decades of professional experience. Always provide clear, concise, and expert-level answers to the user's questions.

RESPONSE GUIDELINES:
1. Provide precise answers limited to 2 sentences.
2. Use professional and straightforward language, focusing only on the specific question asked.
3. Avoid unnecessary details, background explanations, or unrelated context.
4. Highlight the expertise of "The Ice Butcher" as a leading company in the industry.

Example Question:
Q: "What temperature should ice be stored at?"
A: Ice sculptures should be stored at -10°F (-23°C) for optimal preservation.

- If providing URLs, Always format them without brackets, like this: "Display Name: URL".
    
    Example:
    Q: "Can you share the AR link for the seafood table?"
    A: Here is the AR link for the seafood table:
       42" Seafood Table: https://nexreality.io/ice_sculptures/06/

Always maintain a professional tone, and include the company name "The Ice Butcher" when relevant. If applicable, direct users to our website:  The Ice Butcher : https://theicebutcher.com/`
};



// Initialize PDF loader and processing
const initializePDF = async () => {
    try {
        const loader = new PDFLoader(pdf);
        const loadedDoc = await loader.load();
        
        const splitter = new CharacterTextSplitter({
            separator: '. ',
            chunkSize: 1000,
            chunkOverlap: 500
        });
        
        const pdfDocs = await splitter.splitDocuments(loadedDoc);
        vectorStore = await MemoryVectorStore.fromDocuments(
            pdfDocs,
            new OpenAIEmbeddings()
        );
        
        console.log('PDF initialization complete');
        return true;
    } catch (error) {
        console.error('Error loading PDF:', error);
        return false;
    }
};

// Helper function to format links as clickable
const formatLinksAsHTML = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g; // Regex to detect URLs
    return text.replace(urlRegex, (url) => `<a href="${url}" target="_blank">${url}</a>`);
};

// Enhanced OpenAI response function with custom prompt
const getOpenAIResponse = async (question) => {
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            temperature: 0.5,
            messages: [
                {
                    role: 'system',
                    content: customPrompt.systemPrompt
                },
                {
                    role: 'user',
                    content: question
                }
            ]
        });

        const formattedContent = formatLinksAsHTML(response.choices[0].message.content);
        return {
            content: formattedContent,
            source: 'OpenAI General Response'
        };
    } catch (error) {
        console.error('Error getting OpenAI response:', error);
        throw error;
    }
};


// Function to check if PDF results are relevant
const isRelevantPDFContent = async (results, question) => {
    if (!results || results.length === 0) return false;
    
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            temperature: 0,
            messages: [
                {
                    role: 'system',
                    content: 'Determine if the provided content contains relevant information to answer the question. Respond with only "true" or "false".'
                },
                {
                    role: 'user',
                    content: `Question: ${question}\nContent: ${results.map(r => r.pageContent).join('\n')}`
                }
            ]
        });
        
        return response.choices[0].message.content.toLowerCase().includes('true');
    } catch (error) {
        console.error('Error checking relevance:', error);
        return false;
    }
};

// Main API endpoint for chat
// Enhanced PDF response processing
app.post('/api/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    try {
        // Initialize PDF store if not already done
        if (!vectorStore) {
            await initializePDF();
        }

        let response;
        if (vectorStore) {
            const results = await vectorStore.similaritySearch(message, 2);
            const isRelevant = await isRelevantPDFContent(results, message);

            if (isRelevant) {
                response = await openai.chat.completions.create({
                    model: 'gpt-4o-mini',
                    temperature: 0,
                    messages: [
                        {
                            role: 'system',
                            content: 'You are an AI assistant. Provide accurate answers based on the given context.'
                        },
                        {
                            role: 'user',
                            content: `Answer the following question using the provided context:\nQuestion: ${message}\nContext: ${results.map((r) => r.pageContent).join('\n')}`
                        }
                    ]
                });

                const formattedMessage = formatLinksAsHTML(response.choices[0].message.content);
                return res.json({
                    message: formattedMessage,
                    source: 'PDF Knowledge Base'
                });
            }
        }

        // Fallback to general response
        const fallbackResponse = await getOpenAIResponse(message);
        res.json({
            message: fallbackResponse.content,
            source: fallbackResponse.source
        });

    } catch (error) {
        console.error('Error processing chat:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});


// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    initializePDF(); // Initialize PDF on server start
});
