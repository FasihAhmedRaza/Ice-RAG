import { configDotenv } from 'dotenv'
configDotenv()

import OpenAI from 'openai'
export const openai = new OpenAI(process.env.OPENAI_API_KEY)